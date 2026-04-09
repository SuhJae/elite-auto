from __future__ import annotations

from collections import deque
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.actions.navigation_ocr import _find_window_client_region, _resolve_output_region
from app.actions.align_support.models import AlignConfig, CompassMarker, CompassReadResult
from app.actions.starport_buy import build_standalone_context
from app.actions.track_center_reticle import ReticleTrackerConfig, detect_center_reticle
from app.adapters.capture_dxcam import DxcamCapture
from app.config import AppConfig
from app.domain.context import Context
from app.domain.result import Result
from app.domain.protocols import Region


# Edit these values for standalone testing of this file.
STANDALONE_CONFIG_PATH: str | None = None
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_COMPASS_OUTER_RADIUS_PX = 32.0
STANDALONE_COMPASS_INNER_RADIUS_PX = 28.0
STANDALONE_COMPASS_CONTROL_ROTATION_DEGREES = 0.0
STANDALONE_TIMEOUT_SECONDS: float | None = None

WINDOW_TITLE = "Elite Dangerous"



@dataclass(slots=True)
class _AxisControllerState:
    axis: str
    previous_error: float | None = None
    filtered_derivative: float = 0.0
    previous_output: float = 0.0
    last_error_sign: int = 0
    sign_flip_times: deque[float] = field(default_factory=deque)


@dataclass(slots=True)
class _AxisDynamicsProfile:
    axis: str
    px_per_second: float | None = None
    samples: int = 0
    last_active_key: str | None = None
    last_error: float | None = None
    last_timestamp: float | None = None


@dataclass(slots=True)
class _ConsensusSample:
    dx: float
    dy: float
    control_dx: float
    control_dy: float
    distance: float
    marker_state: str


@dataclass(slots=True)
class _NearCenterConsensusState:
    settle_started_at: float | None = None
    sample_started_at: float | None = None
    next_sample_at: float | None = None
    samples: list[_ConsensusSample] = field(default_factory=list)
    cooldown_until: float = -math.inf

    @property
    def is_active(self) -> bool:
        return self.settle_started_at is not None

    @property
    def is_sampling(self) -> bool:
        return self.sample_started_at is not None

    def start(self, now: float) -> None:
        self.settle_started_at = now
        self.sample_started_at = None
        self.next_sample_at = None
        self.samples.clear()

    def begin_sampling(self, now: float) -> None:
        self.sample_started_at = now
        self.next_sample_at = now
        self.samples.clear()

    def reset(self, cooldown_until: float = -math.inf) -> None:
        self.settle_started_at = None
        self.sample_started_at = None
        self.next_sample_at = None
        self.samples.clear()
        self.cooldown_until = cooldown_until


@dataclass(slots=True)
class _AxisSteeringState:
    axis: str
    active_key: str | None = None
    last_press_at: float = -math.inf
    last_release_at: float = -math.inf


@dataclass(slots=True)
class _LoopTelemetry:
    last_frame_at: float | None = None
    loop_fps: float = 0.0
    capture_fps: float = 0.0
    last_capture_at: float | None = None

    def update(self, now: float, capture_completed_at: float) -> tuple[float, float]:
        dt = 0.0
        if self.last_frame_at is not None:
            dt = max(0.0, now - self.last_frame_at)
            if dt > 1e-6:
                instant_fps = 1.0 / dt
                self.loop_fps = instant_fps if self.loop_fps <= 0.0 else ((0.85 * self.loop_fps) + (0.15 * instant_fps))

        if self.last_capture_at is not None:
            capture_dt = max(0.0, capture_completed_at - self.last_capture_at)
            if capture_dt > 1e-6:
                instant_capture_fps = 1.0 / capture_dt
                self.capture_fps = (
                    instant_capture_fps
                    if self.capture_fps <= 0.0
                    else ((0.85 * self.capture_fps) + (0.15 * instant_capture_fps))
                )

        self.last_frame_at = now
        self.last_capture_at = capture_completed_at
        return dt, self.loop_fps


class _SteeringState:
    def __init__(self, input_adapter: Any) -> None:
        self._input_adapter = input_adapter
        self._axes = {
            "pitch": _AxisSteeringState(axis="pitch"),
            "yaw": _AxisSteeringState(axis="yaw"),
        }

    def active_keys(self) -> dict[str, str | None]:
        return {axis: state.active_key for axis, state in self._axes.items()}

    def release_all(self, now: float) -> None:
        for axis in self._axes:
            self._force_release(axis, now)

    def update_axis(
        self,
        axis: str,
        output: float,
        positive_key: str,
        negative_key: str,
        engage_threshold: float,
        release_threshold: float,
        now: float,
        config: AlignConfig,
    ) -> dict[str, Any]:
        state = self._axes[axis]
        desired_key = _desired_key_for_output(
            output=output,
            active_key=state.active_key,
            positive_key=positive_key,
            negative_key=negative_key,
            engage_threshold=engage_threshold,
            release_threshold=release_threshold,
        )
        transition: str | None = None

        if desired_key == state.active_key:
            return {"axis": axis, "active_key": state.active_key, "desired_key": desired_key, "transition": None}

        if state.active_key is not None:
            held_for = now - state.last_press_at
            if held_for >= config.key_state_min_hold_seconds:
                self._input_adapter.key_up(state.active_key)
                state.last_release_at = now
                transition = f"release:{state.active_key}"
                state.active_key = None
            else:
                return {
                    "axis": axis,
                    "active_key": state.active_key,
                    "desired_key": desired_key,
                    "transition": "hold_lock",
                }

        if desired_key is None:
            return {"axis": axis, "active_key": state.active_key, "desired_key": desired_key, "transition": transition}

        released_for = now - state.last_release_at
        if released_for < config.key_state_min_release_seconds:
            return {
                "axis": axis,
                "active_key": state.active_key,
                "desired_key": desired_key,
                "transition": "release_lock" if transition is None else transition,
            }

        self._input_adapter.key_down(desired_key)
        state.active_key = desired_key
        state.last_press_at = now
        transition = f"press:{desired_key}" if transition is None else f"{transition},press:{desired_key}"
        return {"axis": axis, "active_key": state.active_key, "desired_key": desired_key, "transition": transition}

    def _force_release(self, axis: str, now: float) -> None:
        state = self._axes[axis]
        if state.active_key is None:
            return
        self._input_adapter.key_up(state.active_key)
        state.active_key = None
        state.last_release_at = now


def _sign(value: float, *, epsilon: float = 1e-6) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


def _slew_limit(previous_output: float, target_output: float, max_delta: float) -> float:
    if max_delta <= 0.0:
        return previous_output
    lower = previous_output - max_delta
    upper = previous_output + max_delta
    return max(lower, min(upper, target_output))


def _desired_key_for_output(
    output: float,
    active_key: str | None,
    positive_key: str,
    negative_key: str,
    engage_threshold: float,
    release_threshold: float,
) -> str | None:
    if active_key == positive_key:
        if output <= -engage_threshold:
            return negative_key
        if output < release_threshold:
            return None
        return positive_key

    if active_key == negative_key:
        if output >= engage_threshold:
            return positive_key
        if output > -release_threshold:
            return None
        return negative_key

    if output >= engage_threshold:
        return positive_key
    if output <= -engage_threshold:
        return negative_key
    return None


def _compute_axis_controller_output(
    axis: str,
    error_px: float,
    dt: float,
    marker_state: str,
    controller_state: _AxisControllerState,
    dynamics_profile: _AxisDynamicsProfile | None,
    config: AlignConfig,
) -> dict[str, Any]:
    if marker_state == "filled":
        kp = config.filled_kp
        kd = config.filled_kd
        mode = "filled_fine"
    else:
        kp = config.hollow_kp
        kd = config.hollow_kd
        mode = "hollow_coarse"

    gain_scale = _dynamics_gain_scale(axis, dynamics_profile, config)
    kp *= gain_scale
    kd *= gain_scale

    if controller_state.previous_error is None or dt <= 1e-6:
        derivative = 0.0
    else:
        derivative = (error_px - controller_state.previous_error) / dt

    alpha = _clamp(config.error_derivative_low_pass_alpha, 0.0, 1.0)
    filtered_derivative = (
        derivative
        if controller_state.previous_error is None
        else ((alpha * derivative) + ((1.0 - alpha) * controller_state.filtered_derivative))
    )
    target_output = _clamp((kp * error_px) + (kd * filtered_derivative), -1.0, 1.0)
    max_delta = max(0.0, config.controller_output_slew_per_second) * max(dt, 0.0)
    output = target_output if controller_state.previous_error is None else _slew_limit(
        controller_state.previous_output,
        target_output,
        max_delta,
    )

    controller_state.previous_error = error_px
    controller_state.filtered_derivative = filtered_derivative
    controller_state.previous_output = output

    return {
        "axis": axis,
        "mode": mode,
        "error_px": error_px,
        "derivative": filtered_derivative,
        "output": output,
        "kp": kp,
        "kd": kd,
        "gain_scale": gain_scale,
        "profile_px_per_second": None if dynamics_profile is None else dynamics_profile.px_per_second,
    }


def _track_axis_oscillation(
    controller_state: _AxisControllerState,
    error_px: float,
    now: float,
    config: AlignConfig,
) -> bool:
    sign = _sign(error_px)
    if sign != 0 and controller_state.last_error_sign != 0 and sign != controller_state.last_error_sign:
        controller_state.sign_flip_times.append(now)

    controller_state.last_error_sign = sign if sign != 0 else controller_state.last_error_sign
    while (
        controller_state.sign_flip_times
        and (now - controller_state.sign_flip_times[0]) > config.oscillation_sign_flip_window_seconds
    ):
        controller_state.sign_flip_times.popleft()

    return len(controller_state.sign_flip_times) >= config.oscillation_sign_flip_threshold


def _controller_output_for_marker(
    marker: CompassMarker,
    dt: float,
    controller_states: dict[str, _AxisControllerState],
    dynamics_profiles: dict[str, _AxisDynamicsProfile] | None,
    config: AlignConfig,
) -> dict[str, dict[str, Any]]:
    return {
        "yaw": _compute_axis_controller_output(
            axis="yaw",
            error_px=marker.control_dx,
            dt=dt,
            marker_state=marker.marker_state,
            controller_state=controller_states["yaw"],
            dynamics_profile=None if dynamics_profiles is None else dynamics_profiles["yaw"],
            config=config,
        ),
        "pitch": _compute_axis_controller_output(
            axis="pitch",
            error_px=marker.control_dy,
            dt=dt,
            marker_state=marker.marker_state,
            controller_state=controller_states["pitch"],
            dynamics_profile=None if dynamics_profiles is None else dynamics_profiles["pitch"],
            config=config,
        ),
    }


def _controller_output_for_errors(
    *,
    marker_state: str,
    control_dx: float,
    control_dy: float,
    dt: float,
    controller_states: dict[str, _AxisControllerState],
    dynamics_profiles: dict[str, _AxisDynamicsProfile] | None,
    config: AlignConfig,
) -> dict[str, dict[str, Any]]:
    return {
        "yaw": _compute_axis_controller_output(
            axis="yaw",
            error_px=control_dx,
            dt=dt,
            marker_state=marker_state,
            controller_state=controller_states["yaw"],
            dynamics_profile=None if dynamics_profiles is None else dynamics_profiles["yaw"],
            config=config,
        ),
        "pitch": _compute_axis_controller_output(
            axis="pitch",
            error_px=control_dy,
            dt=dt,
            marker_state=marker_state,
            controller_state=controller_states["pitch"],
            dynamics_profile=None if dynamics_profiles is None else dynamics_profiles["pitch"],
            config=config,
        ),
    }


def _dynamics_gain_scale(axis: str, profile: _AxisDynamicsProfile | None, config: AlignConfig) -> float:
    if not config.runtime_profile_enabled or profile is None or profile.px_per_second is None or profile.samples <= 0:
        return 1.0

    reference = (
        config.runtime_profile_yaw_reference_px_per_second
        if axis == "yaw"
        else config.runtime_profile_pitch_reference_px_per_second
    )
    if reference <= 1e-6 or profile.px_per_second <= 1e-6:
        return 1.0

    return _clamp(
        reference / profile.px_per_second,
        config.runtime_profile_gain_scale_min,
        config.runtime_profile_gain_scale_max,
    )


def _update_runtime_dynamics_profile(
    profile: _AxisDynamicsProfile,
    *,
    active_key: str | None,
    error_px: float,
    now: float,
    config: AlignConfig,
) -> None:
    if not config.runtime_profile_enabled:
        return

    if active_key is None:
        profile.last_active_key = None
        profile.last_error = None
        profile.last_timestamp = None
        return

    if profile.last_active_key != active_key or profile.last_timestamp is None or profile.last_error is None:
        profile.last_active_key = active_key
        profile.last_error = error_px
        profile.last_timestamp = now
        return

    dt = max(0.0, now - profile.last_timestamp)
    if dt <= 1e-6:
        return

    improvement = abs(profile.last_error) - abs(error_px)
    if improvement >= config.fine_model_observation_floor_px:
        observed_px_per_second = improvement / dt
        learning_rate = _clamp(config.runtime_profile_learning_rate, 0.0, 1.0)
        profile.px_per_second = (
            observed_px_per_second
            if profile.px_per_second is None
            else ((1.0 - learning_rate) * profile.px_per_second) + (learning_rate * observed_px_per_second)
        )
        profile.samples += 1

    profile.last_error = error_px
    profile.last_timestamp = now


def _is_near_center_candidate(marker: CompassMarker, config: AlignConfig) -> bool:
    return marker.is_filled and marker.distance <= config.near_center_consensus_trigger_distance_px


def _detect_final_reticle(
    frame: np.ndarray,
    config: AlignConfig,
) -> tuple[dict[str, Any], float | None, float | None]:
    tracker_config = ReticleTrackerConfig(
        search_radius_px=max(96, int(config.final_reticle_search_radius_px)),
        capture_target_fps=max(1, int(config.control_target_fps)),
        show_mask_window=False,
    )
    detection, _, _ = detect_center_reticle(frame, tracker_config)
    height, width = frame.shape[:2]
    screen_center_x = width / 2.0
    screen_center_y = height / 2.0

    reticle_debug: dict[str, Any] = {
        "found": detection.found,
        "screen_center_x": round(screen_center_x, 3),
        "screen_center_y": round(screen_center_y, 3),
        "search_region": detection.search_region,
    }
    if detection.found and detection.center_x is not None and detection.center_y is not None:
        dx = detection.center_x - screen_center_x
        dy = detection.center_y - screen_center_y
        reticle_debug.update(
            {
                "center_x": round(detection.center_x, 3),
                "center_y": round(detection.center_y, 3),
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "distance": round(math.hypot(dx, dy), 3),
                "outer_radius_px": (
                    None if detection.outer_radius_px is None else round(detection.outer_radius_px, 3)
                ),
                "score": None if detection.score is None else round(detection.score, 3),
                "metrics": {key: round(value, 3) for key, value in detection.metrics.items()},
            }
        )
        return reticle_debug, dx, dy
    return reticle_debug, None, None




def _record_consensus_sample(
    consensus_state: _NearCenterConsensusState,
    marker: CompassMarker,
    now: float,
    config: AlignConfig,
) -> None:
    if not consensus_state.is_sampling or consensus_state.next_sample_at is None:
        return
    if now + 1e-6 < consensus_state.next_sample_at:
        return

    consensus_state.samples.append(
        _ConsensusSample(
            dx=marker.dx,
            dy=marker.dy,
            control_dx=marker.control_dx,
            control_dy=marker.control_dy,
            distance=marker.distance,
            marker_state=marker.marker_state,
        )
    )
    sample_count = max(1, config.near_center_consensus_samples)
    if len(consensus_state.samples) >= sample_count:
        consensus_state.next_sample_at = math.inf
        return

    if sample_count <= 1:
        consensus_state.next_sample_at = now
        return

    assert consensus_state.sample_started_at is not None
    spacing = max(0.0, config.near_center_consensus_span_seconds) / max(1, sample_count - 1)
    consensus_state.next_sample_at = consensus_state.sample_started_at + (len(consensus_state.samples) * spacing)


def _summarize_consensus(
    samples: list[_ConsensusSample],
    config: AlignConfig,
) -> dict[str, Any] | None:
    if not samples:
        return None

    mean_dx = sum(sample.dx for sample in samples) / len(samples)
    mean_dy = sum(sample.dy for sample in samples) / len(samples)
    mean_control_dx = sum(sample.control_dx for sample in samples) / len(samples)
    mean_control_dy = sum(sample.control_dy for sample in samples) / len(samples)
    mean_distance = math.hypot(mean_dx, mean_dy)
    max_axis_spread = max(
        max(abs(sample.dx - mean_dx), abs(sample.dy - mean_dy))
        for sample in samples
    )
    success = (
        abs(mean_dx) < config.axis_alignment_tolerance_px
        and abs(mean_dy) < config.axis_alignment_tolerance_px
        and mean_distance < config.alignment_tolerance_px
        and max_axis_spread <= config.near_center_consensus_max_axis_spread_px
    )
    return {
        "sample_count": len(samples),
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "mean_control_dx": mean_control_dx,
        "mean_control_dy": mean_control_dy,
        "mean_distance": mean_distance,
        "max_axis_spread_px": max_axis_spread,
        "success": success,
    }

@dataclass(slots=True)
class AlignToTargetCompass:
    """Align the ship to the target compass using pitch and yaw only."""

    config: AlignConfig = field(default_factory=AlignConfig)

    name = "align_to_target_compass"

    def run(self, context: Context) -> Result:
        capture = context.capture
        input_adapter = context.input_adapter
        if capture is None or input_adapter is None:
            return Result.fail("Compass alignment requires both capture and input control.")

        initial_state = context.state_reader.snapshot()
        if initial_state.is_docked:
            return Result.fail(
                "Cannot align to the target compass while docked.",
                debug={"state": initial_state.to_debug_dict()},
            )

        try:
            output_index, capture_region = self._resolve_capture_region(context)
        except RuntimeError as exc:
            return Result.fail(str(exc))

        previous_target_fps: int | None = None
        if isinstance(capture, DxcamCapture):
            previous_target_fps = capture.set_target_fps(self.config.control_target_fps)

        context.logger.info(
            "Starting compass alignment",
            extra={
                "output_index": output_index,
                "capture_region": capture_region,
                "align_config": {
                    "center_x": self.config.center_x,
                    "center_y": self.config.center_y,
                    "roi_size": self.config.roi_size,
                    "control_target_fps": self.config.control_target_fps,
                    "alignment_tolerance_px": self.config.alignment_tolerance_px,
                    "axis_alignment_tolerance_px": self.config.axis_alignment_tolerance_px,
                    "alignment_dwell_seconds": self.config.alignment_dwell_seconds,
                    "final_reticle_enabled": self.config.final_reticle_enabled,
                    "final_reticle_track_anywhere": self.config.final_reticle_track_anywhere,
                    "final_reticle_alignment_tolerance_px": self.config.final_reticle_alignment_tolerance_px,
                    "final_reticle_trigger_distance_px": self.config.final_reticle_trigger_distance_px,
                    "missing_detection_fail_seconds": self.config.missing_detection_fail_seconds,
                    "debug_window_enabled": self.config.debug_window_enabled,
                    "debug_snapshot_interval_seconds": self.config.debug_snapshot_interval_seconds,
                    "filled_kp": self.config.filled_kp,
                    "filled_kd": self.config.filled_kd,
                    "hollow_kp": self.config.hollow_kp,
                    "hollow_kd": self.config.hollow_kd,
                    "pitch_engage_threshold": self.config.pitch_engage_threshold,
                    "pitch_release_threshold": self.config.pitch_release_threshold,
                    "yaw_engage_threshold": self.config.yaw_engage_threshold,
                    "yaw_release_threshold": self.config.yaw_release_threshold,
                    "controller_output_slew_per_second": self.config.controller_output_slew_per_second,
                },
            },
        )

        deadline = None if self.config.timeout_seconds is None else (time.monotonic() + self.config.timeout_seconds)
        controller_states = {
            "pitch": _AxisControllerState(axis="pitch"),
            "yaw": _AxisControllerState(axis="yaw"),
        }
        dynamics_profiles = {
            "pitch": _AxisDynamicsProfile(axis="pitch"),
            "yaw": _AxisDynamicsProfile(axis="yaw"),
        }
        steering = _SteeringState(input_adapter)
        consensus_state = _NearCenterConsensusState()
        telemetry = _LoopTelemetry()
        debug_window_name = "Elite Auto Compass Debug"
        last_status_log_at = 0.0
        missing_started_at: float | None = None
        dwell_started_at: float | None = None
        last_periodic_snapshot_at: float | None = None
        last_anomaly_snapshot_at = -math.inf
        last_debug: dict[str, Any] = {
            "output_index": output_index,
            "capture_region": capture_region,
            "initial_state": initial_state.to_debug_dict(),
            "controller_target_fps": self.config.control_target_fps,
        }
        result: Result | None = None

        try:
            while deadline is None or time.monotonic() < deadline:
                try:
                    frame = self._capture_frame(capture, capture_region)
                except RuntimeError as exc:
                    last_debug["capture_error"] = str(exc)
                    result = Result.fail("Capture failed while aligning to the target compass.", debug=last_debug)
                    break

                capture_completed_at = time.monotonic()
                read_result = detect_compass_marker(frame, self.config)
                now = time.monotonic()
                dt, loop_fps = telemetry.update(now, capture_completed_at)
                last_debug["loop_fps"] = round(loop_fps, 3)
                last_debug["capture_fps"] = round(telemetry.capture_fps, 3)
                last_debug["read_result"] = read_result.to_debug_dict()

                controller_outputs: dict[str, dict[str, Any]] = {
                    "pitch": {"axis": "pitch", "mode": "idle", "error_px": 0.0, "derivative": 0.0, "output": 0.0},
                    "yaw": {"axis": "yaw", "mode": "idle", "error_px": 0.0, "derivative": 0.0, "output": 0.0},
                }
                anomaly_reason: str | None = None
                marker = read_result.marker

                if not read_result.is_detected or marker is None:
                    steering.release_all(now)
                    _reset_controller_states(controller_states)
                    consensus_state.reset()
                    last_debug.pop("final_reticle", None)
                    dwell_started_at = None
                    if missing_started_at is None:
                        missing_started_at = now
                    missing_elapsed = now - missing_started_at
                    last_debug["missing_elapsed_seconds"] = round(missing_elapsed, 3)
                    if missing_elapsed >= self.config.missing_detection_fail_seconds:
                        snapshot_paths = _save_debug_capture_bundle(
                            context,
                            "align_missing_marker",
                            frame=frame,
                            debug_image=frame,
                            config=self.config,
                        )
                        last_debug["debug_snapshot"] = str(snapshot_paths["debug"])
                        last_debug["debug_sources"] = {key: str(value) for key, value in snapshot_paths.items()}
                        result = Result.fail("Compass marker could not be detected reliably.", debug=last_debug)
                        break
                else:
                    missing_started_at = None
                    last_debug.update(marker.to_debug_dict())
                    active_before_update = steering.active_keys()
                    _update_runtime_dynamics_profile(
                        dynamics_profiles["pitch"],
                        active_key=active_before_update["pitch"],
                        error_px=marker.control_dy,
                        now=now,
                        config=self.config,
                    )
                    _update_runtime_dynamics_profile(
                        dynamics_profiles["yaw"],
                        active_key=active_before_update["yaw"],
                        error_px=marker.control_dx,
                        now=now,
                        config=self.config,
                    )

                    reticle_dx: float | None = None
                    reticle_dy: float | None = None
                    last_debug.pop("final_reticle", None)
                    should_try_final_reticle = (
                        self.config.final_reticle_enabled
                        and (
                            self.config.final_reticle_track_anywhere
                            or marker.distance <= self.config.final_reticle_trigger_distance_px
                        )
                    )
                    if should_try_final_reticle:
                        reticle_debug, candidate_reticle_dx, candidate_reticle_dy = _detect_final_reticle(
                            frame,
                            self.config,
                        )
                        reticle_debug["tracking_mode"] = (
                            "anywhere" if self.config.final_reticle_track_anywhere else "near_center_only"
                        )
                        last_debug["final_reticle"] = reticle_debug
                        if candidate_reticle_dx is not None and candidate_reticle_dy is not None:
                            reticle_dx = candidate_reticle_dx
                            reticle_dy = candidate_reticle_dy

                    in_consensus_zone = (
                        not self.config.final_reticle_enabled
                        and
                        self.config.near_center_consensus_enabled
                        and now >= consensus_state.cooldown_until
                        and _is_near_center_candidate(marker, self.config)
                    )

                    if reticle_dx is not None and reticle_dy is not None:
                        consensus_state.reset()
                        control_dx, control_dy = _rotate_offset(
                            reticle_dx,
                            reticle_dy,
                            self.config.compass_control_rotation_radians,
                        )
                        reticle_distance = math.hypot(reticle_dx, reticle_dy)
                        if reticle_distance <= self.config.final_reticle_alignment_tolerance_px:
                            if dwell_started_at is None:
                                dwell_started_at = now
                            dwell_elapsed = now - dwell_started_at
                            last_debug["alignment_dwell_elapsed_seconds"] = round(dwell_elapsed, 3)
                            steering.release_all(now)
                            _reset_controller_states(controller_states)
                            controller_outputs["pitch"] = {
                                "axis": "pitch",
                                "mode": "final_reticle_hold",
                                "error_px": control_dy,
                                "derivative": 0.0,
                                "output": 0.0,
                                "gain_scale": 1.0,
                                "profile_px_per_second": dynamics_profiles["pitch"].px_per_second,
                            }
                            controller_outputs["yaw"] = {
                                "axis": "yaw",
                                "mode": "final_reticle_hold",
                                "error_px": control_dx,
                                "derivative": 0.0,
                                "output": 0.0,
                                "gain_scale": 1.0,
                                "profile_px_per_second": dynamics_profiles["yaw"].px_per_second,
                            }
                            last_debug["axis_transitions"] = {
                                "pitch": {"axis": "pitch", "active_key": None, "desired_key": None, "transition": None},
                                "yaw": {"axis": "yaw", "active_key": None, "desired_key": None, "transition": None},
                            }
                            if dwell_elapsed >= self.config.alignment_dwell_seconds:
                                result = Result.ok("Ship aligned to the target compass.", debug=last_debug)
                                break
                        else:
                            dwell_started_at = None
                            controller_outputs = _controller_output_for_errors(
                                marker_state="filled",
                                control_dx=control_dx,
                                control_dy=control_dy,
                                dt=dt,
                                controller_states=controller_states,
                                dynamics_profiles=dynamics_profiles,
                                config=self.config,
                            )
                            controller_outputs["yaw"]["mode"] = "final_reticle"
                            controller_outputs["pitch"]["mode"] = "final_reticle"
                            pitch_update = steering.update_axis(
                                axis="pitch",
                                output=controller_outputs["pitch"]["output"],
                                positive_key=context.config.controls.pitch_down,
                                negative_key=context.config.controls.pitch_up,
                                engage_threshold=self.config.pitch_engage_threshold,
                                release_threshold=self.config.pitch_release_threshold,
                                now=now,
                                config=self.config,
                            )
                            yaw_update = steering.update_axis(
                                axis="yaw",
                                output=controller_outputs["yaw"]["output"],
                                positive_key=context.config.controls.yaw_right,
                                negative_key=context.config.controls.yaw_left,
                                engage_threshold=self.config.yaw_engage_threshold,
                                release_threshold=self.config.yaw_release_threshold,
                                now=now,
                                config=self.config,
                            )
                            last_debug["axis_transitions"] = {"pitch": pitch_update, "yaw": yaw_update}
                    elif in_consensus_zone:
                        steering.release_all(now)
                        _reset_controller_states(controller_states)
                        dwell_started_at = None
                        if not consensus_state.is_active:
                            consensus_state.start(now)
                        elif not consensus_state.is_sampling:
                            if (now - consensus_state.settle_started_at) >= self.config.near_center_consensus_pause_seconds:
                                consensus_state.begin_sampling(now)
                        else:
                            _record_consensus_sample(consensus_state, marker, now, self.config)
                            if len(consensus_state.samples) >= max(1, self.config.near_center_consensus_samples):
                                consensus_summary = _summarize_consensus(consensus_state.samples, self.config)
                                if consensus_summary is not None:
                                    last_debug["near_center_consensus"] = {
                                        "sample_count": consensus_summary["sample_count"],
                                        "mean_dx": round(consensus_summary["mean_dx"], 3),
                                        "mean_dy": round(consensus_summary["mean_dy"], 3),
                                        "mean_distance": round(consensus_summary["mean_distance"], 3),
                                        "max_axis_spread_px": round(consensus_summary["max_axis_spread_px"], 3),
                                        "success": consensus_summary["success"],
                                    }
                                    if consensus_summary["success"]:
                                        result = Result.ok("Ship aligned to the target compass.", debug=last_debug)
                                        break
                                    controller_outputs = _controller_output_for_errors(
                                        marker_state=marker.marker_state,
                                        control_dx=consensus_summary["mean_control_dx"],
                                        control_dy=consensus_summary["mean_control_dy"],
                                        dt=dt,
                                        controller_states=controller_states,
                                        dynamics_profiles=dynamics_profiles,
                                        config=self.config,
                                    )
                                    controller_outputs["yaw"]["mode"] = "filled_consensus_nudge"
                                    controller_outputs["pitch"]["mode"] = "filled_consensus_nudge"
                                    consensus_state.reset(cooldown_until=now + self.config.near_center_consensus_cooldown_seconds)

                        if result is None and any(values["output"] != 0.0 for values in controller_outputs.values()):
                            pitch_update = steering.update_axis(
                                axis="pitch",
                                output=controller_outputs["pitch"]["output"],
                                positive_key=context.config.controls.pitch_down,
                                negative_key=context.config.controls.pitch_up,
                                engage_threshold=self.config.pitch_engage_threshold,
                                release_threshold=self.config.pitch_release_threshold,
                                now=now,
                                config=self.config,
                            )
                            yaw_update = steering.update_axis(
                                axis="yaw",
                                output=controller_outputs["yaw"]["output"],
                                positive_key=context.config.controls.yaw_right,
                                negative_key=context.config.controls.yaw_left,
                                engage_threshold=self.config.yaw_engage_threshold,
                                release_threshold=self.config.yaw_release_threshold,
                                now=now,
                                config=self.config,
                            )
                            last_debug["axis_transitions"] = {"pitch": pitch_update, "yaw": yaw_update}
                        else:
                            last_debug["alignment_dwell_elapsed_seconds"] = round(
                                max(0.0, now - consensus_state.settle_started_at),
                                3,
                            )
                            last_debug["axis_transitions"] = {
                                "pitch": {"axis": "pitch", "active_key": None, "desired_key": None, "transition": None},
                                "yaw": {"axis": "yaw", "active_key": None, "desired_key": None, "transition": None},
                            }
                            controller_outputs["pitch"] = {
                                "axis": "pitch",
                                "mode": "near_center_pause" if not consensus_state.is_sampling else "near_center_sampling",
                                "error_px": marker.control_dy,
                                "derivative": 0.0,
                                "output": 0.0,
                                "gain_scale": 1.0,
                                "profile_px_per_second": dynamics_profiles["pitch"].px_per_second,
                            }
                            controller_outputs["yaw"] = {
                                "axis": "yaw",
                                "mode": "near_center_pause" if not consensus_state.is_sampling else "near_center_sampling",
                                "error_px": marker.control_dx,
                                "derivative": 0.0,
                                "output": 0.0,
                                "gain_scale": 1.0,
                                "profile_px_per_second": dynamics_profiles["yaw"].px_per_second,
                            }
                    else:
                        if consensus_state.is_active:
                            anomaly_reason = "dwell_exit"
                            consensus_state.reset(cooldown_until=now + self.config.near_center_consensus_cooldown_seconds)

                        if marker.is_aligned(self.config):
                            if dwell_started_at is None:
                                dwell_started_at = now
                            dwell_elapsed = now - dwell_started_at
                            last_debug["alignment_dwell_elapsed_seconds"] = round(dwell_elapsed, 3)
                            steering.release_all(now)
                            _reset_controller_states(controller_states)
                            if dwell_elapsed >= self.config.alignment_dwell_seconds:
                                result = Result.ok("Ship aligned to the target compass.", debug=last_debug)
                                break
                        else:
                            if dwell_started_at is not None:
                                anomaly_reason = "dwell_exit"
                            dwell_started_at = None

                            if marker.is_hollow and marker.distance <= self.config.center_tolerance_px:
                                controller_outputs["pitch"] = {
                                    "axis": "pitch",
                                    "mode": "breakout",
                                    "error_px": marker.control_dy,
                                    "derivative": 0.0,
                                    "output": -1.0,
                                    "gain_scale": 1.0,
                                    "profile_px_per_second": dynamics_profiles["pitch"].px_per_second,
                                }
                                controller_outputs["yaw"] = {
                                    "axis": "yaw",
                                    "mode": "breakout",
                                    "error_px": marker.control_dx,
                                    "derivative": 0.0,
                                    "output": 0.0,
                                    "gain_scale": 1.0,
                                    "profile_px_per_second": dynamics_profiles["yaw"].px_per_second,
                                }
                                controller_states["pitch"].previous_output = -1.0
                                controller_states["yaw"].previous_output = 0.0
                            else:
                                controller_outputs = _controller_output_for_marker(
                                    marker,
                                    dt,
                                    controller_states,
                                    dynamics_profiles,
                                    self.config,
                                )
                                pitch_oscillation = _track_axis_oscillation(
                                    controller_states["pitch"],
                                    marker.control_dy,
                                    now,
                                    self.config,
                                )
                                yaw_oscillation = _track_axis_oscillation(
                                    controller_states["yaw"],
                                    marker.control_dx,
                                    now,
                                    self.config,
                                )
                                if pitch_oscillation or yaw_oscillation:
                                    anomaly_reason = "oscillation"

                            pitch_update = steering.update_axis(
                                axis="pitch",
                                output=controller_outputs["pitch"]["output"],
                                positive_key=context.config.controls.pitch_down,
                                negative_key=context.config.controls.pitch_up,
                                engage_threshold=self.config.pitch_engage_threshold,
                                release_threshold=self.config.pitch_release_threshold,
                                now=now,
                                config=self.config,
                            )
                            yaw_update = steering.update_axis(
                                axis="yaw",
                                output=controller_outputs["yaw"]["output"],
                                positive_key=context.config.controls.yaw_right,
                                negative_key=context.config.controls.yaw_left,
                                engage_threshold=self.config.yaw_engage_threshold,
                                release_threshold=self.config.yaw_release_threshold,
                                now=now,
                                config=self.config,
                            )
                            last_debug["axis_transitions"] = {"pitch": pitch_update, "yaw": yaw_update}

                dwell_elapsed = 0.0 if dwell_started_at is None else max(0.0, now - dwell_started_at)
                active_keys = steering.active_keys()
                last_debug["controller_outputs"] = {
                    axis: {
                        "mode": values["mode"],
                        "error_px": round(values["error_px"], 3),
                        "derivative": round(values["derivative"], 3),
                        "output": round(values["output"], 3),
                        "gain_scale": round(values.get("gain_scale", 1.0), 3),
                        "profile_px_per_second": (
                            None
                            if values.get("profile_px_per_second") is None
                            else round(values["profile_px_per_second"], 3)
                        ),
                    }
                    for axis, values in controller_outputs.items()
                }
                last_debug["active_keys"] = active_keys
                last_debug["runtime_dynamics"] = {
                    axis: {
                        "px_per_second": (
                            None if profile.px_per_second is None else round(profile.px_per_second, 3)
                        ),
                        "samples": profile.samples,
                    }
                    for axis, profile in dynamics_profiles.items()
                }

                debug_frame = _build_alignment_debug_frame(
                    frame=frame,
                    marker=marker,
                    config=self.config,
                    controller_outputs=controller_outputs,
                    active_keys=active_keys,
                    dwell_elapsed=dwell_elapsed,
                    loop_fps=telemetry.loop_fps,
                    capture_fps=telemetry.capture_fps,
                    anomaly_reason=anomaly_reason,
                )

                if self.config.debug_window_enabled:
                    cv2.imshow(debug_window_name, debug_frame)
                    cv2.waitKey(1)

                if (
                    self.config.debug_window_enabled
                    and self.config.debug_snapshot_interval_seconds > 0.0
                    and (
                        last_periodic_snapshot_at is None
                        or (now - last_periodic_snapshot_at) >= self.config.debug_snapshot_interval_seconds
                    )
                ):
                    snapshot_paths = _save_debug_capture_bundle(
                        context,
                        "align_periodic",
                        frame=frame,
                        debug_image=debug_frame,
                        config=self.config,
                    )
                    last_debug["periodic_snapshot"] = str(snapshot_paths["debug"])
                    last_debug["periodic_sources"] = {key: str(value) for key, value in snapshot_paths.items()}
                    last_periodic_snapshot_at = now

                if anomaly_reason is not None and (now - last_anomaly_snapshot_at) >= 1.0:
                    snapshot_paths = _save_debug_capture_bundle(
                        context,
                        f"align_{anomaly_reason}",
                        frame=frame,
                        debug_image=debug_frame,
                        config=self.config,
                    )
                    snapshot_path = snapshot_paths["debug"]
                    last_debug["anomaly_snapshot"] = str(snapshot_path)
                    last_debug["anomaly_sources"] = {key: str(value) for key, value in snapshot_paths.items()}
                    last_anomaly_snapshot_at = now
                    context.logger.info(
                        "Compass alignment anomaly",
                        extra={
                            "reason": anomaly_reason,
                            "snapshot_path": str(snapshot_path),
                        },
                    )

                if (now - last_status_log_at) >= 0.5:
                    context.logger.info(
                        "Compass alignment loop",
                        extra={
                            "detected": marker is not None,
                            "status": read_result.status,
                            "loop_fps": round(telemetry.loop_fps, 2),
                            "capture_fps": round(telemetry.capture_fps, 2),
                            "dwell_elapsed_seconds": round(dwell_elapsed, 3),
                            "active_keys": active_keys,
                            "controller_outputs": last_debug["controller_outputs"],
                        },
                    )
                    last_status_log_at = now

            if result is None:
                timeout_frame = self._capture_frame(capture, capture_region)
                timeout_debug_frame = _build_alignment_debug_frame(
                    frame=timeout_frame,
                    marker=None,
                    config=self.config,
                    controller_outputs={
                        "pitch": {"axis": "pitch", "mode": "timeout", "error_px": 0.0, "derivative": 0.0, "output": 0.0},
                        "yaw": {"axis": "yaw", "mode": "timeout", "error_px": 0.0, "derivative": 0.0, "output": 0.0},
                    },
                    active_keys=steering.active_keys(),
                    dwell_elapsed=0.0,
                    loop_fps=telemetry.loop_fps,
                    capture_fps=telemetry.capture_fps,
                    anomaly_reason="timeout",
                )
                snapshot_paths = _save_debug_capture_bundle(
                    context,
                    "align_timeout",
                    frame=frame,
                    debug_image=timeout_debug_frame,
                    config=self.config,
                )
                last_debug["debug_snapshot"] = str(snapshot_paths["debug"])
                last_debug["debug_sources"] = {key: str(value) for key, value in snapshot_paths.items()}
                last_debug["timeout_seconds"] = self.config.timeout_seconds
                result = Result.fail("Timed out while aligning to the target compass.", debug=last_debug)
        finally:
            steering.release_all(time.monotonic())
            if self.config.debug_window_enabled:
                try:
                    cv2.destroyWindow(debug_window_name)
                except cv2.error:
                    pass
            if isinstance(capture, DxcamCapture):
                capture.set_target_fps(previous_target_fps)

        assert result is not None
        return result

    def _resolve_capture_region(self, context: Context) -> tuple[int, Region]:
        capture = context.capture
        if capture is None:
            raise RuntimeError("Capture is not available.")

        window_capture_region = _find_window_client_region(WINDOW_TITLE)
        if window_capture_region is None:
            raise RuntimeError(f"Could not locate the {WINDOW_TITLE!r} window.")

        output_index, output_relative_region = _resolve_output_region(window_capture_region)
        if output_index is None or output_relative_region is None:
            raise RuntimeError("Could not map the Elite window onto a dxcam output.")

        if output_relative_region[2] < self.config.min_width or output_relative_region[3] < self.config.min_height:
            raise RuntimeError(
                "Compass alignment currently supports only the calibrated 1920x1080 HUD layout."
            )

        if isinstance(capture, DxcamCapture):
            capture.set_output_index(output_index)

        return output_index, output_relative_region

    def _capture_frame(self, capture: Any, capture_region: Region) -> Any:
        last_error: RuntimeError | None = None
        attempts = max(1, self.config.capture_retry_attempts)
        for attempt in range(1, attempts + 1):
            try:
                frame = capture.grab(region=capture_region)
            except RuntimeError as exc:
                last_error = exc
                frame = None
            if frame is not None:
                return frame
            if attempt < attempts:
                time.sleep(self.config.capture_retry_interval_seconds)

        if last_error is not None:
            raise RuntimeError(str(last_error))
        raise RuntimeError("Capture returned no frame.")


from app.actions.align_support.detection import _extract_roi, _rotate_offset, detect_compass_marker


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _reset_controller_states(controller_states: dict[str, _AxisControllerState]) -> None:
    for state in controller_states.values():
        state.previous_error = None
        state.filtered_derivative = 0.0
        state.previous_output = 0.0


def _build_alignment_debug_frame(
    frame: np.ndarray,
    marker: CompassMarker | None,
    config: AlignConfig,
    controller_outputs: dict[str, dict[str, Any]],
    active_keys: dict[str, str | None],
    dwell_elapsed: float,
    loop_fps: float,
    capture_fps: float,
    anomaly_reason: str | None,
) -> np.ndarray:
    roi = _extract_roi(frame, config.roi_region()).copy()
    if marker is not None and marker.compass_center_x is not None and marker.compass_center_y is not None:
        center_x = int(round(marker.compass_center_x - config.roi_region()[0]))
        center_y = int(round(marker.compass_center_y - config.roi_region()[1]))
        detected_radius = (
            max(1, int(round(marker.compass_radius_estimate_px)))
            if marker.compass_radius_estimate_px is not None
            else int(round(config.compass_radius_px))
        )
    else:
        center_x = roi.shape[1] // 2
        center_y = roi.shape[0] // 2
        detected_radius = int(round(config.compass_radius_px))
    inner_anchor_radius = max(1, int(round(detected_radius - config.inner_ring_outer_radius_offset_px)))
    strict_radius = max(1, int(round(config.alignment_tolerance_px)))
    breakout_radius = max(1, int(round(config.center_tolerance_px)))

    cv2.drawMarker(roi, (center_x, center_y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
    cv2.circle(roi, (center_x, center_y), inner_anchor_radius, (255, 0, 0), 1)
    cv2.circle(roi, (center_x, center_y), detected_radius, (0, 0, 255), 1)
    cv2.circle(roi, (center_x, center_y), strict_radius, (0, 220, 0), 1)
    cv2.circle(roi, (center_x, center_y), breakout_radius, (0, 160, 255), 1)

    if marker is not None:
        marker_local_x = int(round(marker.marker_x - config.roi_region()[0]))
        marker_local_y = int(round(marker.marker_y - config.roi_region()[1]))
        marker_color = (0, 255, 0) if marker.is_filled else (0, 200, 255)
        cv2.circle(roi, (marker_local_x, marker_local_y), 3, marker_color, -1)
        cv2.line(roi, (center_x, center_y), (marker_local_x, marker_local_y), marker_color, 1)

    scale = max(1, int(config.debug_window_scale))
    debug_frame = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    lines = [
        f"loop_fps={loop_fps:.1f} capture_fps={capture_fps:.1f}",
        f"marker={marker.marker_state if marker is not None else 'missing'} dwell={dwell_elapsed:.2f}s",
        f"center=({center_x},{center_y}) inner_r={inner_anchor_radius} outer_r={detected_radius}",
        "rings: inner=blue outer=red",
        (
            "pitch "
            f"err={controller_outputs['pitch']['error_px']:.2f} "
            f"out={controller_outputs['pitch']['output']:.2f} "
            f"gain={controller_outputs['pitch'].get('gain_scale', 1.0):.2f} "
            f"key={active_keys['pitch'] or '-'}"
        ),
        (
            "yaw   "
            f"err={controller_outputs['yaw']['error_px']:.2f} "
            f"out={controller_outputs['yaw']['output']:.2f} "
            f"gain={controller_outputs['yaw'].get('gain_scale', 1.0):.2f} "
            f"key={active_keys['yaw'] or '-'}"
        ),
    ]
    pitch_profile = controller_outputs["pitch"].get("profile_px_per_second")
    yaw_profile = controller_outputs["yaw"].get("profile_px_per_second")
    if pitch_profile is not None or yaw_profile is not None:
        lines.append(
            "profile "
            f"pitch={('-' if pitch_profile is None else f'{pitch_profile:.1f}')}"
            f" yaw={('-' if yaw_profile is None else f'{yaw_profile:.1f}')}"
        )
    if anomaly_reason is not None:
        lines.append(f"anomaly={anomaly_reason}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 18
    for line in lines:
        cv2.putText(debug_frame, line, (8, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

    return debug_frame


def _save_debug_snapshot(context: Context, name: str, image: Any) -> Path:
    if context.vision is not None:
        return context.vision.save_debug_snapshot(name, image)

    context.debug_snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = context.debug_snapshot_dir / f"{timestamp}_{name}.png"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write debug snapshot: {path}")
    return path


def _save_debug_capture_bundle(
    context: Context,
    name: str,
    *,
    frame: np.ndarray,
    debug_image: np.ndarray,
    config: AlignConfig,
) -> dict[str, Path]:
    def write_image(suffix: str, image: np.ndarray) -> Path:
        snapshot_name = f"{name}_{suffix}"
        if context.vision is not None:
            return context.vision.save_debug_snapshot(snapshot_name, image)

        context.debug_snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = context.debug_snapshot_dir / f"{timestamp}_{snapshot_name}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write debug snapshot: {path}")
        return path

    compass_roi = _extract_region(frame, config.roi_region())
    reticle_region = _center_search_region(
        width=frame.shape[1],
        height=frame.shape[0],
        search_radius=max(96, int(config.final_reticle_search_radius_px)),
    )
    reticle_crop = _extract_region(frame, reticle_region)

    return {
        "debug": write_image("debug", debug_image),
        "full": write_image("full", frame),
        "compass_roi": write_image("compass_roi", compass_roi),
        "reticle_roi": write_image("reticle_roi", reticle_crop),
    }


def _extract_region(image: np.ndarray, region: Region) -> np.ndarray:
    x, y, width, height = region
    left = max(0, int(x))
    top = max(0, int(y))
    right = min(image.shape[1], left + max(1, int(width)))
    bottom = min(image.shape[0], top + max(1, int(height)))
    return image[top:bottom, left:right].copy()


def _center_search_region(*, width: int, height: int, search_radius: int) -> Region:
    center_x = width // 2
    center_y = height // 2
    half = max(32, int(search_radius))
    left = max(0, center_x - half)
    top = max(0, center_y - half)
    right = min(width, center_x + half)
    bottom = min(height, center_y + half)
    return (left, top, right - left, bottom - top)

def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    context = build_standalone_context(config)

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(
            f"Warning: focusing game window. Starting align_to_target_compass in {STANDALONE_START_DELAY_SECONDS:.1f} seconds..."
        )
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    action = AlignToTargetCompass(
        config=AlignConfig(
            compass_radius_px=STANDALONE_COMPASS_OUTER_RADIUS_PX,
            inner_ring_outer_radius_offset_px=max(
                0.0,
                STANDALONE_COMPASS_OUTER_RADIUS_PX - STANDALONE_COMPASS_INNER_RADIUS_PX,
            ),
            compass_control_rotation_degrees=STANDALONE_COMPASS_CONTROL_ROTATION_DEGREES,
            debug_window_enabled=True,
            final_reticle_enabled=True,
            near_center_consensus_pause_seconds=2.0,
            near_center_consensus_samples=3,
            near_center_consensus_span_seconds=0.30,
            timeout_seconds=STANDALONE_TIMEOUT_SECONDS,
        )
    )
    result = action.run(context)
    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
