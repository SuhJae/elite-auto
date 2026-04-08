from __future__ import annotations

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
from app.actions.starport_buy import build_standalone_context
from app.adapters.capture_dxcam import DxcamCapture
from app.config import AppConfig
from app.domain.context import Context
from app.domain.result import Result
from app.domain.protocols import Region


# Edit these values for standalone testing of this file.
STANDALONE_CONFIG_PATH: str | None = None
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_PITCH_ADJUSTMENT_SCALE = 1.0
STANDALONE_YAW_ADJUSTMENT_SCALE = 1.0
STANDALONE_COMPASS_RADIUS_PX = 32.0
STANDALONE_COMPASS_CONTROL_ROTATION_DEGREES = 0.0
STANDALONE_MAX_PULSE_SECONDS = 8.0
STANDALONE_SETTLE_SECONDS_AFTER_INPUT: float | None = None
STANDALONE_TIMEOUT_SECONDS: float | None = None
STANDALONE_IDLE_READ_BACKOFF_SECONDS: float | None = None

WINDOW_TITLE = "Elite Dangerous"


@dataclass(slots=True)
class AlignConfig:
    center_x: int = 731
    center_y: int = 818
    roi_size: int = 80
    compass_radius_px: float = 32.0
    compass_control_rotation_degrees: float = 0.0
    center_tolerance_px: int = 4
    alignment_tolerance_px: float = 2.0
    axis_alignment_tolerance_px: float = 1.0
    confirmation_axis_tolerance_px: float = 5.0
    confirmation_distance_tolerance_px: float = 6.0
    settle_seconds_after_input: float = 2.0
    idle_read_backoff_seconds: float = 0.25
    confirmation_reads: int = 3
    timeout_seconds: float | None = None
    pitch_rate_deg_per_sec: float = 18.20
    yaw_rate_deg_per_sec: float = 10.11
    response_ramp_seconds: float = 0.8
    pitch_adjustment_scale: float = 1.0
    yaw_adjustment_scale: float = 1.0
    center_breakout_degrees: float = 12.0
    filled_micro_error_px: float = 8.0
    filled_soft_pitch_error_px: float = 18.0
    filled_soft_yaw_error_px: float = 20.0
    filled_micro_scale: float = 0.25
    filled_soft_scale: float = 0.50
    filled_micro_pitch_max_seconds: float = 0.30
    filled_micro_yaw_max_seconds: float = 0.60
    filled_soft_pitch_max_seconds: float = 1.25
    filled_soft_yaw_max_seconds: float = 2.00
    overshoot_damping_factor: float = 0.50
    catastrophic_overshoot_damping_factor: float = 0.35
    catastrophic_overshoot_error_growth_px: float = 3.0
    catastrophic_overshoot_error_growth_ratio: float = 1.50
    catastrophic_overshoot_max_pulse_seconds: float = 0.75
    fine_model_distance_px: float = 15.0
    fine_axis_dominance_ratio: float = 1.25
    ultra_fine_distance_px: float = 4.0
    ultra_fine_target_axis_px: float = 0.25
    ultra_fine_cross_guard_px: float = 2.25
    ultra_fine_cross_coeff_scale: float = 0.35
    fine_pitch_primary_delta_coeff: float = 18.5
    fine_pitch_cross_delta_coeff: float = 22.0
    fine_yaw_primary_delta_coeff: float = 12.5
    fine_yaw_cross_delta_coeff: float = 3.0
    fine_model_learning_rate: float = 0.20
    fine_model_observation_floor_px: float = 0.35
    fine_model_min_effective_learning_rate: float = 0.35
    fine_model_min_effective_boost: float = 1.25
    fine_model_primary_coeff_min: float = 4.0
    fine_model_primary_coeff_max: float = 40.0
    fine_model_cross_coeff_min: float = 0.0
    fine_model_cross_coeff_max: float = 30.0
    fine_model_cross_update_max_ratio: float = 1.25
    fine_model_min_samples_for_full_pulse: int = 3
    fine_pitch_min_seconds: float = 0.10
    fine_pitch_max_seconds: float = 0.50
    fine_yaw_min_seconds: float = 0.12
    fine_yaw_max_seconds: float = 0.90
    fine_pitch_uncalibrated_max_seconds: float = 0.35
    fine_yaw_uncalibrated_max_seconds: float = 0.55
    ultra_fine_pitch_min_seconds: float = 0.05
    ultra_fine_pitch_max_seconds: float = 0.25
    ultra_fine_yaw_min_seconds: float = 0.06
    ultra_fine_yaw_max_seconds: float = 0.35
    post_input_extra_settle_seconds_max: float = 0.8
    min_width: int = 1920
    min_height: int = 1080
    warm_h_min: int = 5
    warm_h_max: int = 45
    warm_s_min: int = 15
    warm_v_min: int = 110
    fallback_warm_s_min: int = 40
    fallback_warm_v_min: int = 100
    pale_h_max: int = 110
    pale_s_max: int = 90
    pale_v_min: int = 165
    pale_red_min: int = 130
    pale_green_min: int = 150
    pale_blue_red_delta_max: int = 50
    pale_green_red_delta_max: int = 60
    marker_min_area: int = 3
    marker_max_area: int = 80
    max_marker_distance_px: float = 40.0
    inner_disk_radius_px: int = 3
    outer_disk_radius_px: int = 6
    filled_inner_occupancy_threshold: float = 0.30
    definitive_filled_inner_occupancy_threshold: float = 0.80
    refinement_search_radius_px: int = 3
    max_consecutive_missing_or_ambiguous_reads: int = 20
    max_consecutive_ambiguous_reads: int = 2
    max_pulse_seconds: float = 8.0
    capture_retry_attempts: int = 5
    capture_retry_interval_seconds: float = 0.10
    hollow_edge_phase_epsilon_degrees: float = 1.0
    hollow_edge_push_degrees: float = 6.0

    def roi_region(self) -> Region:
        half = self.roi_size // 2
        return (self.center_x - half, self.center_y - half, self.roi_size, self.roi_size)

    @property
    def compass_control_rotation_radians(self) -> float:
        return math.radians(self.compass_control_rotation_degrees)


@dataclass(slots=True)
class CompassMarker:
    marker_state: str
    marker_x: float
    marker_y: float
    dx: float
    dy: float
    control_dx: float
    control_dy: float
    distance: float
    normalized_radius: float
    front_semisphere_radians: float
    target_off_boresight_radians: float
    phase_adjustment_radians: float
    component_area: int
    inner_occupancy: float
    outer_ring_occupancy: float
    roi_region: Region

    @property
    def is_filled(self) -> bool:
        return self.marker_state == "filled"

    @property
    def is_hollow(self) -> bool:
        return self.marker_state == "hollow"

    def is_aligned(self, config: AlignConfig) -> bool:
        return (
            self.is_filled
            and abs(self.dx) < config.axis_alignment_tolerance_px
            and abs(self.dy) < config.axis_alignment_tolerance_px
            and self.distance < config.alignment_tolerance_px
        )

    def is_confirmation_aligned(self, config: AlignConfig) -> bool:
        return (
            self.is_filled
            and abs(self.dx) < config.confirmation_axis_tolerance_px
            and abs(self.dy) < config.confirmation_axis_tolerance_px
            and self.distance < config.confirmation_distance_tolerance_px
        )

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "marker_state": self.marker_state,
            "marker_x": round(self.marker_x, 3),
            "marker_y": round(self.marker_y, 3),
            "relative_x": round(self.dx, 3),
            "relative_y": round(self.dy, 3),
            "screen_relative_x": round(self.dx, 3),
            "screen_relative_y": round(self.dy, 3),
            "control_relative_x": round(self.control_dx, 3),
            "control_relative_y": round(self.control_dy, 3),
            "distance": round(self.distance, 3),
            "normalized_radius": round(self.normalized_radius, 4),
            "front_semisphere_radians": round(self.front_semisphere_radians, 4),
            "front_semisphere_degrees": round(math.degrees(self.front_semisphere_radians), 3),
            "target_off_boresight_radians": round(self.target_off_boresight_radians, 4),
            "target_off_boresight_degrees": round(math.degrees(self.target_off_boresight_radians), 3),
            "phase_adjustment_radians": round(self.phase_adjustment_radians, 4),
            "phase_adjustment_degrees": round(math.degrees(self.phase_adjustment_radians), 3),
            "component_area": self.component_area,
            "inner_occupancy": round(self.inner_occupancy, 4),
            "outer_ring_occupancy": round(self.outer_ring_occupancy, 4),
            "roi_region": self.roi_region,
        }


@dataclass(slots=True)
class CompassReadResult:
    status: str
    marker: CompassMarker | None = None

    @property
    def is_detected(self) -> bool:
        return self.status in {"filled", "hollow"} and self.marker is not None

    def to_debug_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.marker is not None:
            payload["marker"] = self.marker.to_debug_dict()
        return payload


@dataclass(slots=True)
class PulseEstimate:
    axis: str
    axis_error_px: float
    axis_error_radians: float
    axis_error_degrees: float
    target_hold_seconds: float
    axis_rate_deg_per_sec: float
    response_ramp_seconds: float
    ramp_angle_degrees: float
    adjustment_scale: float
    unclamped_seconds: float
    pulse_seconds: float
    pulse_mode: str = "angle"
    damping_factor: float = 1.0
    response_primary_coeff: float | None = None
    response_cross_coeff: float | None = None
    target_axis_px: float | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        payload = {
            "axis": self.axis,
            "axis_error_px": round(self.axis_error_px, 3),
            "axis_error_radians": round(self.axis_error_radians, 4),
            "axis_error_degrees": round(self.axis_error_degrees, 3),
            "target_hold_seconds": round(self.target_hold_seconds, 4),
            "axis_rate_deg_per_sec": round(self.axis_rate_deg_per_sec, 4),
            "response_ramp_seconds": round(self.response_ramp_seconds, 4),
            "ramp_angle_degrees": round(self.ramp_angle_degrees, 3),
            "adjustment_scale": round(self.adjustment_scale, 4),
            "unclamped_seconds": round(self.unclamped_seconds, 4),
            "pulse_seconds": round(self.pulse_seconds, 4),
            "pulse_mode": self.pulse_mode,
            "damping_factor": round(self.damping_factor, 4),
        }
        if self.response_primary_coeff is not None:
            payload["response_primary_coeff"] = round(self.response_primary_coeff, 4)
        if self.response_cross_coeff is not None:
            payload["response_cross_coeff"] = round(self.response_cross_coeff, 4)
        if self.target_axis_px is not None:
            payload["target_axis_px"] = round(self.target_axis_px, 4)
        return payload


@dataclass(slots=True)
class _AxisResponseModel:
    axis: str
    primary_coeff: float
    cross_coeff: float
    minimum_effective_seconds: float = 0.0
    samples: int = 0

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "primary_coeff": round(self.primary_coeff, 4),
            "cross_coeff": round(self.cross_coeff, 4),
            "minimum_effective_seconds": round(self.minimum_effective_seconds, 4),
            "samples": self.samples,
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

        context.logger.info(
            "Starting compass alignment",
            extra={
                "output_index": output_index,
                "capture_region": capture_region,
                "align_config": {
                    "center_x": self.config.center_x,
                    "center_y": self.config.center_y,
                    "roi_size": self.config.roi_size,
                    "compass_radius_px": self.config.compass_radius_px,
                    "compass_control_rotation_degrees": self.config.compass_control_rotation_degrees,
                    "center_tolerance_px": self.config.center_tolerance_px,
                    "alignment_tolerance_px": self.config.alignment_tolerance_px,
                    "axis_alignment_tolerance_px": self.config.axis_alignment_tolerance_px,
                    "settle_seconds_after_input": self.config.settle_seconds_after_input,
                    "idle_read_backoff_seconds": self.config.idle_read_backoff_seconds,
                    "response_ramp_seconds": self.config.response_ramp_seconds,
                    "pitch_adjustment_scale": self.config.pitch_adjustment_scale,
                    "yaw_adjustment_scale": self.config.yaw_adjustment_scale,
                    "filled_micro_error_px": self.config.filled_micro_error_px,
                    "filled_soft_pitch_error_px": self.config.filled_soft_pitch_error_px,
                    "filled_soft_yaw_error_px": self.config.filled_soft_yaw_error_px,
                    "filled_micro_scale": self.config.filled_micro_scale,
                    "filled_soft_scale": self.config.filled_soft_scale,
                    "filled_micro_pitch_max_seconds": self.config.filled_micro_pitch_max_seconds,
                    "filled_micro_yaw_max_seconds": self.config.filled_micro_yaw_max_seconds,
                    "filled_soft_pitch_max_seconds": self.config.filled_soft_pitch_max_seconds,
                    "filled_soft_yaw_max_seconds": self.config.filled_soft_yaw_max_seconds,
                    "overshoot_damping_factor": self.config.overshoot_damping_factor,
                    "catastrophic_overshoot_damping_factor": self.config.catastrophic_overshoot_damping_factor,
                    "catastrophic_overshoot_error_growth_px": self.config.catastrophic_overshoot_error_growth_px,
                    "catastrophic_overshoot_error_growth_ratio": self.config.catastrophic_overshoot_error_growth_ratio,
                    "catastrophic_overshoot_max_pulse_seconds": self.config.catastrophic_overshoot_max_pulse_seconds,
                    "fine_model_distance_px": self.config.fine_model_distance_px,
                    "fine_axis_dominance_ratio": self.config.fine_axis_dominance_ratio,
                    "ultra_fine_distance_px": self.config.ultra_fine_distance_px,
                    "ultra_fine_target_axis_px": self.config.ultra_fine_target_axis_px,
                    "ultra_fine_cross_guard_px": self.config.ultra_fine_cross_guard_px,
                    "ultra_fine_cross_coeff_scale": self.config.ultra_fine_cross_coeff_scale,
                    "fine_pitch_primary_delta_coeff": self.config.fine_pitch_primary_delta_coeff,
                    "fine_pitch_cross_delta_coeff": self.config.fine_pitch_cross_delta_coeff,
                    "fine_yaw_primary_delta_coeff": self.config.fine_yaw_primary_delta_coeff,
                    "fine_yaw_cross_delta_coeff": self.config.fine_yaw_cross_delta_coeff,
                    "fine_model_learning_rate": self.config.fine_model_learning_rate,
                    "fine_model_min_effective_learning_rate": self.config.fine_model_min_effective_learning_rate,
                    "fine_model_min_effective_boost": self.config.fine_model_min_effective_boost,
                    "fine_model_cross_update_max_ratio": self.config.fine_model_cross_update_max_ratio,
                    "fine_model_min_samples_for_full_pulse": self.config.fine_model_min_samples_for_full_pulse,
                    "fine_pitch_min_seconds": self.config.fine_pitch_min_seconds,
                    "fine_pitch_max_seconds": self.config.fine_pitch_max_seconds,
                    "fine_yaw_min_seconds": self.config.fine_yaw_min_seconds,
                    "fine_yaw_max_seconds": self.config.fine_yaw_max_seconds,
                    "fine_pitch_uncalibrated_max_seconds": self.config.fine_pitch_uncalibrated_max_seconds,
                    "fine_yaw_uncalibrated_max_seconds": self.config.fine_yaw_uncalibrated_max_seconds,
                    "ultra_fine_pitch_min_seconds": self.config.ultra_fine_pitch_min_seconds,
                    "ultra_fine_pitch_max_seconds": self.config.ultra_fine_pitch_max_seconds,
                    "ultra_fine_yaw_min_seconds": self.config.ultra_fine_yaw_min_seconds,
                    "ultra_fine_yaw_max_seconds": self.config.ultra_fine_yaw_max_seconds,
                    "max_pulse_seconds": self.config.max_pulse_seconds,
                    "capture_retry_attempts": self.config.capture_retry_attempts,
                    "capture_retry_interval_seconds": self.config.capture_retry_interval_seconds,
                    "hollow_edge_phase_epsilon_degrees": self.config.hollow_edge_phase_epsilon_degrees,
                    "hollow_edge_push_degrees": self.config.hollow_edge_push_degrees,
                },
            },
        )

        deadline = None if self.config.timeout_seconds is None else (time.monotonic() + self.config.timeout_seconds)
        consecutive_missing_or_ambiguous = 0
        consecutive_ambiguous = 0
        confirmation_count = 0
        response_models = _build_response_models(self.config)
        last_command: _PreviousCommand | None = None
        last_debug: dict[str, Any] = {
            "output_index": output_index,
            "capture_region": capture_region,
            "initial_state": initial_state.to_debug_dict(),
            "response_models": {
                axis: model.to_debug_dict() for axis, model in response_models.items()
            },
        }

        while deadline is None or time.monotonic() < deadline:
            try:
                frame = self._capture_frame(capture, capture_region)
            except RuntimeError as exc:
                last_debug["capture_error"] = str(exc)
                return Result.fail("Capture failed while aligning to the target compass.", debug=last_debug)
            read_result = detect_compass_marker(frame, self.config)
            last_debug["read_result"] = read_result.to_debug_dict()

            if not read_result.is_detected:
                context.logger.info(
                    "Compass marker read",
                    extra={
                        "status": read_result.status,
                        "detected": False,
                    },
                )
                consecutive_missing_or_ambiguous += 1
                if read_result.status == "ambiguous":
                    consecutive_ambiguous += 1
                    if consecutive_ambiguous >= self.config.max_consecutive_ambiguous_reads:
                        snapshot_path = _save_debug_snapshot(context, "align_ambiguous", frame)
                        last_debug["debug_snapshot"] = str(snapshot_path)
                        return Result.fail("Compass marker classification remained ambiguous.", debug=last_debug)
                else:
                    consecutive_ambiguous = 0

                if consecutive_missing_or_ambiguous >= self.config.max_consecutive_missing_or_ambiguous_reads:
                    snapshot_path = _save_debug_snapshot(context, "align_missing_marker", frame)
                    last_debug["debug_snapshot"] = str(snapshot_path)
                    return Result.fail("Compass marker could not be detected reliably.", debug=last_debug)
                time.sleep(self.config.idle_read_backoff_seconds)
                continue

            marker = read_result.marker
            assert marker is not None
            consecutive_missing_or_ambiguous = 0
            consecutive_ambiguous = 0
            last_debug.update(marker.to_debug_dict())
            response_model_update = _update_response_models(
                marker,
                last_command,
                response_models,
                self.config,
            )
            if response_model_update is not None:
                last_debug["response_model_update"] = response_model_update
                last_debug["response_models"] = {
                    axis: model.to_debug_dict() for axis, model in response_models.items()
                }
            overshoot_feedback = _overshoot_feedback(marker, last_command, self.config)
            if overshoot_feedback is not None:
                last_debug["overshoot_feedback"] = overshoot_feedback
            context.logger.info(
                "Compass marker read",
                extra={
                    "status": marker.marker_state,
                    "detected": True,
                    "screen_relative_x": round(marker.dx, 3),
                    "screen_relative_y": round(marker.dy, 3),
                    "control_relative_x": round(marker.control_dx, 3),
                    "control_relative_y": round(marker.control_dy, 3),
                    "absolute_x": round(marker.marker_x, 3),
                    "absolute_y": round(marker.marker_y, 3),
                    "distance": round(marker.distance, 3),
                    "normalized_radius": round(marker.normalized_radius, 4),
                    "target_off_boresight_degrees": round(math.degrees(marker.target_off_boresight_radians), 3),
                    "phase_adjustment_degrees": round(math.degrees(marker.phase_adjustment_radians), 3),
                    "inner_occupancy": round(marker.inner_occupancy, 4),
                    "outer_ring_occupancy": round(marker.outer_ring_occupancy, 4),
                    "aligned": marker.is_aligned(self.config),
                    "overshoot_feedback": overshoot_feedback,
                    "response_model_update": response_model_update,
                    "response_models": {
                        axis: model.to_debug_dict() for axis, model in response_models.items()
                    },
                },
            )

            strict_alignment = marker.is_aligned(self.config)
            confirmation_alignment = strict_alignment or (
                confirmation_count > 0 and marker.is_confirmation_aligned(self.config)
            )

            if confirmation_alignment:
                confirmation_count += 1
                last_debug["confirmation_count"] = confirmation_count
                context.logger.info(
                    "Compass alignment confirmation",
                    extra={
                        "confirmation_count": confirmation_count,
                        "required_confirmation_reads": self.config.confirmation_reads,
                        "strict_alignment": strict_alignment,
                        "confirmation_alignment": confirmation_alignment,
                        "screen_relative_x": round(marker.dx, 3),
                        "screen_relative_y": round(marker.dy, 3),
                        "control_relative_x": round(marker.control_dx, 3),
                        "control_relative_y": round(marker.control_dy, 3),
                    },
                )
                if confirmation_count >= self.config.confirmation_reads:
                    return Result.ok(
                        "Ship aligned to the target compass.",
                        debug=last_debug,
                    )
                last_command = None
                time.sleep(self.config.settle_seconds_after_input)
                continue

            confirmation_count = 0
            axis, key_name, pulse_estimate = _select_alignment_command(
                marker,
                self.config,
                last_command=last_command,
                response_models=response_models,
            )
            key = getattr(context.config.controls, key_name)
            last_debug.update(
                {
                    "selected_axis": axis,
                    "selected_key_name": key_name,
                    "selected_key": key,
                    "pulse_estimate": pulse_estimate.to_debug_dict(),
                }
            )
            context.logger.info(
                "Compass alignment command",
                extra={
                    "marker_state": marker.marker_state,
                    "screen_relative_x": round(marker.dx, 3),
                    "screen_relative_y": round(marker.dy, 3),
                    "control_relative_x": round(marker.control_dx, 3),
                    "control_relative_y": round(marker.control_dy, 3),
                    "selected_axis": axis,
                    "selected_key_name": key_name,
                    "selected_key": key,
                    "pulse_estimate": pulse_estimate.to_debug_dict(),
                    "response_models": {
                        model_axis: model.to_debug_dict() for model_axis, model in response_models.items()
                    },
                },
            )
            input_adapter.hold(key, pulse_estimate.pulse_seconds)
            last_command = _PreviousCommand(
                axis=axis,
                signed_axis_error=_signed_axis_error(marker, axis),
                pulse_seconds=pulse_estimate.pulse_seconds,
                marker_dx=marker.dx,
                marker_dy=marker.dy,
                pulse_mode=pulse_estimate.pulse_mode,
            )
            time.sleep(_settle_after_input_seconds(pulse_estimate.pulse_seconds, self.config))

        timeout_frame = self._capture_frame(capture, capture_region)
        snapshot_path = _save_debug_snapshot(context, "align_timeout", timeout_frame)
        last_debug["debug_snapshot"] = str(snapshot_path)
        last_debug["timeout_seconds"] = self.config.timeout_seconds
        return Result.fail("Timed out while aligning to the target compass.", debug=last_debug)

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


def detect_compass_marker(image: Any, config: AlignConfig) -> CompassReadResult:
    if image is None:
        return CompassReadResult(status="missing")

    roi = _extract_roi(image, config.roi_region())
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _build_marker_mask(roi, hsv, config)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    candidates = _candidate_components(mask_closed, hsv, config)
    if not candidates:
        fallback_mask = _build_fallback_warm_mask(hsv, config)
        candidates = _candidate_components(fallback_mask, hsv, config)
        if not candidates:
            return CompassReadResult(status="missing")
        mask = fallback_mask

    candidate = candidates[0]
    inner_occupancy, outer_ring_occupancy = _occupancy_scores(
        mask,
        candidate["centroid_x"],
        candidate["centroid_y"],
        config,
    )

    marker_local_x = candidate["centroid_x"]
    marker_local_y = candidate["centroid_y"]

    if inner_occupancy >= config.filled_inner_occupancy_threshold:
        if inner_occupancy >= config.definitive_filled_inner_occupancy_threshold:
            state = "filled"
        else:
            refined = _refine_hollow_center(mask, candidate["centroid_x"], candidate["centroid_y"], config)
            if refined is not None:
                marker_local_x = refined["x"]
                marker_local_y = refined["y"]
                inner_occupancy = refined["inner_occupancy"]
                outer_ring_occupancy = refined["outer_ring_occupancy"]
                state = "hollow"
            else:
                state = "filled"
    elif outer_ring_occupancy > 0.0 and inner_occupancy < config.filled_inner_occupancy_threshold:
        state = "hollow"
    else:
        return CompassReadResult(status="ambiguous")

    marker_x = config.roi_region()[0] + marker_local_x
    marker_y = config.roi_region()[1] + marker_local_y
    dx = marker_x - config.center_x
    dy = marker_y - config.center_y
    control_dx, control_dy = _rotate_offset(dx, dy, config.compass_control_rotation_radians)
    distance = math.hypot(dx, dy)
    normalized_radius = min(1.0, distance / max(config.compass_radius_px, 1.0))
    front_semisphere_radians = math.asin(normalized_radius)
    if state == "filled":
        target_off_boresight_radians = front_semisphere_radians
        phase_adjustment_radians = front_semisphere_radians
    else:
        target_off_boresight_radians = math.pi - front_semisphere_radians
        phase_adjustment_radians = max(0.0, (math.pi / 2.0) - front_semisphere_radians)

    marker = CompassMarker(
        marker_state=state,
        marker_x=marker_x,
        marker_y=marker_y,
        dx=dx,
        dy=dy,
        control_dx=control_dx,
        control_dy=control_dy,
        distance=distance,
        normalized_radius=normalized_radius,
        front_semisphere_radians=front_semisphere_radians,
        target_off_boresight_radians=target_off_boresight_radians,
        phase_adjustment_radians=phase_adjustment_radians,
        component_area=int(candidate["area"]),
        inner_occupancy=inner_occupancy,
        outer_ring_occupancy=outer_ring_occupancy,
        roi_region=config.roi_region(),
    )
    return CompassReadResult(status=state, marker=marker)


def _build_marker_mask(roi: np.ndarray, hsv: np.ndarray, config: AlignConfig) -> np.ndarray:
    warm_mask = cv2.inRange(
        hsv,
        np.array([config.warm_h_min, config.warm_s_min, config.warm_v_min], dtype=np.uint8),
        np.array([config.warm_h_max, 255, 255], dtype=np.uint8),
    )

    blue_channel, green_channel, red_channel = cv2.split(roi)
    blue_minus_red = blue_channel.astype(np.int16) - red_channel.astype(np.int16)
    green_minus_red = green_channel.astype(np.int16) - red_channel.astype(np.int16)
    pale_mask = cv2.inRange(
        hsv,
        np.array([0, 0, config.pale_v_min], dtype=np.uint8),
        np.array([config.pale_h_max, config.pale_s_max, 255], dtype=np.uint8),
    )
    pale_mask = np.logical_and(
        pale_mask > 0,
        np.logical_and.reduce(
            (
                red_channel >= config.pale_red_min,
                green_channel >= config.pale_green_min,
                blue_minus_red <= config.pale_blue_red_delta_max,
                green_minus_red <= config.pale_green_red_delta_max,
            )
        ),
    )

    return np.where(np.logical_or(warm_mask > 0, pale_mask), 255, 0).astype(np.uint8)


def _build_fallback_warm_mask(hsv: np.ndarray, config: AlignConfig) -> np.ndarray:
    """Fallback for sunlit scenes where the primary warm/pale mask over-merges into glare blobs."""

    return cv2.inRange(
        hsv,
        np.array([0, config.fallback_warm_s_min, config.fallback_warm_v_min], dtype=np.uint8),
        np.array([config.warm_h_max, 255, 255], dtype=np.uint8),
    )


def _candidate_components(mask: np.ndarray, hsv: np.ndarray, config: AlignConfig) -> list[dict[str, float]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    local_center_x = config.roi_size / 2.0
    local_center_y = config.roi_size / 2.0
    candidates: list[dict[str, float]] = []

    for index in range(1, num_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < config.marker_min_area or area > config.marker_max_area:
            continue

        centroid_x = float(centroids[index][0])
        centroid_y = float(centroids[index][1])
        if math.hypot(centroid_x - local_center_x, centroid_y - local_center_y) > config.max_marker_distance_px:
            continue

        component_mask = labels == index
        mean_value = float(hsv[:, :, 2][component_mask].mean())
        candidates.append(
            {
                "area": float(area),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "mean_value": mean_value,
            }
        )

    candidates.sort(key=lambda candidate: (candidate["area"], candidate["mean_value"]), reverse=True)
    return candidates


def _occupancy_scores(
    mask: np.ndarray,
    centroid_x: float,
    centroid_y: float,
    config: AlignConfig,
) -> tuple[float, float]:
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    squared_distance = (xx - centroid_x) ** 2 + (yy - centroid_y) ** 2

    inner_mask = squared_distance <= config.inner_disk_radius_px**2
    outer_mask = squared_distance <= config.outer_disk_radius_px**2
    outer_ring_mask = np.logical_and(outer_mask, np.logical_not(inner_mask))

    inner_pixels = mask[inner_mask] > 0
    outer_ring_pixels = mask[outer_ring_mask] > 0

    inner_occupancy = float(inner_pixels.mean()) if inner_pixels.size else 0.0
    outer_ring_occupancy = float(outer_ring_pixels.mean()) if outer_ring_pixels.size else 0.0
    return inner_occupancy, outer_ring_occupancy


def _refine_hollow_center(
    mask: np.ndarray,
    centroid_x: float,
    centroid_y: float,
    config: AlignConfig,
) -> dict[str, float] | None:
    best_candidate: dict[str, float] | None = None
    for offset_x in range(-config.refinement_search_radius_px, config.refinement_search_radius_px + 1):
        for offset_y in range(-config.refinement_search_radius_px, config.refinement_search_radius_px + 1):
            candidate_x = centroid_x + offset_x
            candidate_y = centroid_y + offset_y
            inner_occupancy, outer_ring_occupancy = _occupancy_scores(mask, candidate_x, candidate_y, config)
            if inner_occupancy >= config.filled_inner_occupancy_threshold or outer_ring_occupancy <= 0.0:
                continue

            score = outer_ring_occupancy - inner_occupancy
            candidate = {
                "score": score,
                "x": candidate_x,
                "y": candidate_y,
                "inner_occupancy": inner_occupancy,
                "outer_ring_occupancy": outer_ring_occupancy,
            }
            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

    return best_candidate


def _extract_roi(image: Any, region: Region) -> np.ndarray:
    x, y, width, height = region
    if image.shape[1] < x + width or image.shape[0] < y + height:
        raise ValueError("Image does not match the calibrated ROI bounds for compass detection.")
    return image[y : y + height, x : x + width]


def _rotate_offset(dx: float, dy: float, rotation_radians: float) -> tuple[float, float]:
    cosine = math.cos(rotation_radians)
    sine = math.sin(rotation_radians)
    return (cosine * dx) - (sine * dy), (sine * dx) + (cosine * dy)


def _settle_after_input_seconds(pulse_seconds: float, config: AlignConfig) -> float:
    extra_settle = min(
        config.post_input_extra_settle_seconds_max,
        max(0.0, pulse_seconds - config.response_ramp_seconds),
    )
    return config.settle_seconds_after_input + extra_settle


@dataclass(slots=True)
class _PreviousCommand:
    axis: str
    signed_axis_error: float
    pulse_seconds: float
    marker_dx: float
    marker_dy: float
    pulse_mode: str


def _build_response_models(config: AlignConfig) -> dict[str, _AxisResponseModel]:
    return {
        "pitch": _AxisResponseModel(
            axis="pitch",
            primary_coeff=config.fine_pitch_primary_delta_coeff,
            cross_coeff=config.fine_pitch_cross_delta_coeff,
        ),
        "yaw": _AxisResponseModel(
            axis="yaw",
            primary_coeff=config.fine_yaw_primary_delta_coeff,
            cross_coeff=config.fine_yaw_cross_delta_coeff,
        ),
    }


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _signed_axis_error(marker: CompassMarker, axis: str) -> float:
    return marker.dx if axis == "yaw" else marker.dy


def _is_catastrophic_overshoot(
    marker: CompassMarker,
    last_command: _PreviousCommand | None,
    config: AlignConfig,
) -> bool:
    if last_command is None:
        return False

    current_error = _signed_axis_error(marker, last_command.axis)
    current_magnitude = abs(current_error)
    previous_magnitude = abs(last_command.signed_axis_error)
    error_growth = current_magnitude - previous_magnitude
    if marker.is_hollow and last_command.pulse_mode.startswith("filled"):
        return True

    return (
        error_growth >= config.catastrophic_overshoot_error_growth_px
        and current_magnitude >= (previous_magnitude * config.catastrophic_overshoot_error_growth_ratio)
    )


def _overshoot_feedback(
    marker: CompassMarker,
    last_command: _PreviousCommand | None,
    config: AlignConfig,
) -> dict[str, Any] | None:
    if last_command is None:
        return None

    current_error = _signed_axis_error(marker, last_command.axis)
    sign_flipped = last_command.signed_axis_error * current_error < 0.0
    error_growth = abs(current_error) - abs(last_command.signed_axis_error)
    magnitude_reduction = abs(last_command.signed_axis_error) - abs(current_error)
    return {
        "axis": last_command.axis,
        "previous_signed_error": round(last_command.signed_axis_error, 3),
        "current_signed_error": round(current_error, 3),
        "sign_flipped": sign_flipped,
        "error_growth": round(error_growth, 3),
        "magnitude_reduction": round(magnitude_reduction, 3),
        "last_pulse_seconds": round(last_command.pulse_seconds, 4),
        "catastrophic": _is_catastrophic_overshoot(marker, last_command, config),
    }


def _update_response_models(
    marker: CompassMarker,
    last_command: _PreviousCommand | None,
    response_models: dict[str, _AxisResponseModel],
    config: AlignConfig,
) -> dict[str, Any] | None:
    if last_command is None or not marker.is_filled:
        return None
    if not last_command.pulse_mode.startswith("filled"):
        return None

    axis = last_command.axis
    model = response_models[axis]
    pulse_squared = last_command.pulse_seconds**2
    if pulse_squared <= 1e-6:
        return None

    if axis == "yaw":
        previous_primary = abs(last_command.marker_dx)
        current_primary = abs(marker.dx)
        previous_cross = abs(last_command.marker_dy)
        current_cross = abs(marker.dy)
    else:
        previous_primary = abs(last_command.marker_dy)
        current_primary = abs(marker.dy)
        previous_cross = abs(last_command.marker_dx)
        current_cross = abs(marker.dx)

    observed_primary_delta = previous_primary - current_primary
    observed_cross_delta = current_cross - previous_cross
    info: dict[str, Any] = {
        "axis": axis,
        "previous_primary_error_px": round(previous_primary, 3),
        "current_primary_error_px": round(current_primary, 3),
        "previous_cross_error_px": round(previous_cross, 3),
        "current_cross_error_px": round(current_cross, 3),
        "observed_primary_delta_px": round(observed_primary_delta, 3),
        "observed_cross_delta_px": round(observed_cross_delta, 3),
        "pulse_seconds": round(last_command.pulse_seconds, 4),
    }

    learning_rate = config.fine_model_learning_rate
    min_effective_learning_rate = config.fine_model_min_effective_learning_rate
    primary_coeff_before = model.primary_coeff
    cross_coeff_before = model.cross_coeff
    minimum_effective_seconds_before = model.minimum_effective_seconds

    if observed_primary_delta >= config.fine_model_observation_floor_px:
        observed_primary_coeff = max(0.0, observed_primary_delta) / pulse_squared
        model.primary_coeff = _clamp(
            ((1.0 - learning_rate) * model.primary_coeff) + (learning_rate * observed_primary_coeff),
            config.fine_model_primary_coeff_min,
            config.fine_model_primary_coeff_max,
        )
        model.samples += 1
        info["observed_primary_coeff"] = round(observed_primary_coeff, 4)
        if model.minimum_effective_seconds > 0.0 and last_command.pulse_seconds <= model.minimum_effective_seconds:
            model.minimum_effective_seconds = max(0.0, model.minimum_effective_seconds * 0.95)
    elif observed_primary_delta <= -config.fine_model_observation_floor_px:
        observed_primary_coeff = 0.0
        model.primary_coeff = _clamp(
            ((1.0 - learning_rate) * model.primary_coeff) + (learning_rate * observed_primary_coeff),
            config.fine_model_primary_coeff_min,
            config.fine_model_primary_coeff_max,
        )
        info["observed_primary_coeff"] = round(observed_primary_coeff, 4)

    cross_update_limit = max(
        config.fine_model_observation_floor_px,
        observed_primary_delta * config.fine_model_cross_update_max_ratio,
    )
    if (
        observed_primary_delta >= config.fine_model_observation_floor_px
        and observed_cross_delta >= config.fine_model_observation_floor_px
        and observed_cross_delta <= cross_update_limit
    ):
        observed_cross_coeff = max(0.0, observed_cross_delta) / pulse_squared
        model.cross_coeff = _clamp(
            ((1.0 - learning_rate) * model.cross_coeff) + (learning_rate * observed_cross_coeff),
            config.fine_model_cross_coeff_min,
            config.fine_model_cross_coeff_max,
        )
        info["observed_cross_coeff"] = round(observed_cross_coeff, 4)

    if 0.0 <= observed_primary_delta < config.fine_model_observation_floor_px:
        boosted_floor = last_command.pulse_seconds * config.fine_model_min_effective_boost
        model.minimum_effective_seconds = max(
            model.minimum_effective_seconds,
            ((1.0 - min_effective_learning_rate) * model.minimum_effective_seconds)
            + (min_effective_learning_rate * boosted_floor),
        )

    info["primary_coeff_before"] = round(primary_coeff_before, 4)
    info["primary_coeff_after"] = round(model.primary_coeff, 4)
    info["cross_coeff_before"] = round(cross_coeff_before, 4)
    info["cross_coeff_after"] = round(model.cross_coeff, 4)
    info["minimum_effective_seconds_before"] = round(minimum_effective_seconds_before, 4)
    info["minimum_effective_seconds_after"] = round(model.minimum_effective_seconds, 4)
    info["samples"] = model.samples
    return info


def _select_alignment_command(
    marker: CompassMarker,
    config: AlignConfig,
    last_command: _PreviousCommand | None = None,
    response_models: dict[str, _AxisResponseModel] | None = None,
) -> tuple[str, str, PulseEstimate]:
    if marker.is_hollow and marker.distance <= config.center_tolerance_px:
        axis = "pitch"
        key = "pitch_up"
        pulse_estimate = _estimate_pulse_seconds(
            axis=axis,
            axis_error_px=config.center_tolerance_px + 1,
            axis_error_radians=math.radians(config.center_breakout_degrees),
            config=config,
        )
        return axis, key, pulse_estimate

    if _should_use_fine_model(marker, config):
        axis, key, pulse_estimate = _select_fine_alignment_command(marker, config, response_models)
        pulse_estimate = _apply_overshoot_damping(marker, pulse_estimate, last_command, config)
        return axis, key, pulse_estimate

    if abs(marker.dx) >= abs(marker.dy):
        axis = "yaw"
        key = "yaw_left" if marker.dx < 0 else "yaw_right"
        axis_error_px = abs(marker.dx)
    else:
        axis = "pitch"
        key = "pitch_up" if marker.dy < 0 else "pitch_down"
        axis_error_px = abs(marker.dy)

    axis_error_radians = _axis_error_radians(marker, axis, config)

    pulse_estimate = _estimate_pulse_seconds(
        axis=axis,
        axis_error_px=axis_error_px,
        axis_error_radians=axis_error_radians,
        config=config,
    )
    pulse_estimate = _tune_filled_pulse(marker, pulse_estimate, config)
    pulse_estimate = _apply_overshoot_damping(marker, pulse_estimate, last_command, config)
    return axis, key, pulse_estimate


def _should_use_fine_model(marker: CompassMarker, config: AlignConfig) -> bool:
    return marker.is_filled and marker.distance <= config.fine_model_distance_px


def _select_fine_alignment_command(
    marker: CompassMarker,
    config: AlignConfig,
    response_models: dict[str, _AxisResponseModel] | None = None,
) -> tuple[str, str, PulseEstimate]:
    models = response_models or _build_response_models(config)
    x_error = abs(marker.dx)
    y_error = abs(marker.dy)
    axis_tolerance = config.axis_alignment_tolerance_px

    if x_error <= axis_tolerance and y_error > axis_tolerance:
        candidate = _solve_axis_candidate(
            axis="pitch",
            primary_error_px=y_error,
            cross_error_px=x_error,
            axis_error_radians=_axis_error_radians(marker, "pitch", config),
            axis_rate_deg_per_sec=config.pitch_rate_deg_per_sec,
            response_model=models["pitch"],
            min_seconds=(
                config.ultra_fine_pitch_min_seconds
                if marker.distance <= config.ultra_fine_distance_px
                else config.fine_pitch_min_seconds
            ),
            max_seconds=(
                config.ultra_fine_pitch_max_seconds
                if marker.distance <= config.ultra_fine_distance_px
                else config.fine_pitch_max_seconds
            ),
            target_axis_px=(
                config.ultra_fine_target_axis_px
                if marker.distance <= config.ultra_fine_distance_px
                else config.axis_alignment_tolerance_px
            ),
            pulse_mode=(
                "filled_ultra_fine_model" if marker.distance <= config.ultra_fine_distance_px else "filled_fine_model"
            ),
            config=config,
        )
        return "pitch", "pitch_up" if marker.dy < 0 else "pitch_down", candidate["estimate"]

    if y_error <= axis_tolerance and x_error > axis_tolerance:
        candidate = _solve_axis_candidate(
            axis="yaw",
            primary_error_px=x_error,
            cross_error_px=y_error,
            axis_error_radians=_axis_error_radians(marker, "yaw", config),
            axis_rate_deg_per_sec=config.yaw_rate_deg_per_sec,
            response_model=models["yaw"],
            min_seconds=(
                config.ultra_fine_yaw_min_seconds
                if marker.distance <= config.ultra_fine_distance_px
                else config.fine_yaw_min_seconds
            ),
            max_seconds=(
                config.ultra_fine_yaw_max_seconds
                if marker.distance <= config.ultra_fine_distance_px
                else config.fine_yaw_max_seconds
            ),
            target_axis_px=(
                config.ultra_fine_target_axis_px
                if marker.distance <= config.ultra_fine_distance_px
                else config.axis_alignment_tolerance_px
            ),
            pulse_mode=(
                "filled_ultra_fine_model" if marker.distance <= config.ultra_fine_distance_px else "filled_fine_model"
            ),
            config=config,
        )
        return "yaw", "yaw_left" if marker.dx < 0 else "yaw_right", candidate["estimate"]

    if y_error > (x_error * config.fine_axis_dominance_ratio):
        candidate = _solve_axis_candidate(
            axis="pitch",
            primary_error_px=y_error,
            cross_error_px=x_error,
            axis_error_radians=_axis_error_radians(marker, "pitch", config),
            axis_rate_deg_per_sec=config.pitch_rate_deg_per_sec,
            response_model=models["pitch"],
            min_seconds=(
                config.ultra_fine_pitch_min_seconds
                if marker.distance <= config.ultra_fine_distance_px
                else config.fine_pitch_min_seconds
            ),
            max_seconds=(
                config.ultra_fine_pitch_max_seconds
                if marker.distance <= config.ultra_fine_distance_px
                else config.fine_pitch_max_seconds
            ),
            target_axis_px=(
                config.ultra_fine_target_axis_px
                if marker.distance <= config.ultra_fine_distance_px
                else config.axis_alignment_tolerance_px
            ),
            pulse_mode=(
                "filled_ultra_fine_model" if marker.distance <= config.ultra_fine_distance_px else "filled_fine_model"
            ),
            config=config,
        )
        return "pitch", "pitch_up" if marker.dy < 0 else "pitch_down", candidate["estimate"]

    if x_error > (y_error * config.fine_axis_dominance_ratio):
        candidate = _solve_axis_candidate(
            axis="yaw",
            primary_error_px=x_error,
            cross_error_px=y_error,
            axis_error_radians=_axis_error_radians(marker, "yaw", config),
            axis_rate_deg_per_sec=config.yaw_rate_deg_per_sec,
            response_model=models["yaw"],
            min_seconds=(
                config.ultra_fine_yaw_min_seconds
                if marker.distance <= config.ultra_fine_distance_px
                else config.fine_yaw_min_seconds
            ),
            max_seconds=(
                config.ultra_fine_yaw_max_seconds
                if marker.distance <= config.ultra_fine_distance_px
                else config.fine_yaw_max_seconds
            ),
            target_axis_px=(
                config.ultra_fine_target_axis_px
                if marker.distance <= config.ultra_fine_distance_px
                else config.axis_alignment_tolerance_px
            ),
            pulse_mode=(
                "filled_ultra_fine_model" if marker.distance <= config.ultra_fine_distance_px else "filled_fine_model"
            ),
            config=config,
        )
        return "yaw", "yaw_left" if marker.dx < 0 else "yaw_right", candidate["estimate"]

    use_ultra_fine = marker.distance <= config.ultra_fine_distance_px
    if use_ultra_fine:
        yaw_candidate = _solve_axis_candidate(
            axis="yaw",
            primary_error_px=x_error,
            cross_error_px=y_error,
            axis_error_radians=_axis_error_radians(marker, "yaw", config),
            axis_rate_deg_per_sec=config.yaw_rate_deg_per_sec,
            response_model=models["yaw"],
            min_seconds=config.ultra_fine_yaw_min_seconds,
            max_seconds=config.ultra_fine_yaw_max_seconds,
            target_axis_px=config.ultra_fine_target_axis_px,
            pulse_mode="filled_ultra_fine_model",
            config=config,
        )
        pitch_candidate = _solve_axis_candidate(
            axis="pitch",
            primary_error_px=y_error,
            cross_error_px=x_error,
            axis_error_radians=_axis_error_radians(marker, "pitch", config),
            axis_rate_deg_per_sec=config.pitch_rate_deg_per_sec,
            response_model=models["pitch"],
            min_seconds=config.ultra_fine_pitch_min_seconds,
            max_seconds=config.ultra_fine_pitch_max_seconds,
            target_axis_px=config.ultra_fine_target_axis_px,
            pulse_mode="filled_ultra_fine_model",
            config=config,
        )
    else:
        yaw_candidate = _solve_axis_candidate(
            axis="yaw",
            primary_error_px=x_error,
            cross_error_px=y_error,
            axis_error_radians=_axis_error_radians(marker, "yaw", config),
            axis_rate_deg_per_sec=config.yaw_rate_deg_per_sec,
            response_model=models["yaw"],
            min_seconds=config.fine_yaw_min_seconds,
            max_seconds=config.fine_yaw_max_seconds,
            target_axis_px=config.axis_alignment_tolerance_px,
            pulse_mode="filled_fine_model",
            config=config,
        )
        pitch_candidate = _solve_axis_candidate(
            axis="pitch",
            primary_error_px=y_error,
            cross_error_px=x_error,
            axis_error_radians=_axis_error_radians(marker, "pitch", config),
            axis_rate_deg_per_sec=config.pitch_rate_deg_per_sec,
            response_model=models["pitch"],
            min_seconds=config.fine_pitch_min_seconds,
            max_seconds=config.fine_pitch_max_seconds,
            target_axis_px=config.axis_alignment_tolerance_px,
            pulse_mode="filled_fine_model",
            config=config,
        )

    current_norm = math.hypot(x_error, y_error)
    yaw_norm = yaw_candidate["predicted_norm"]
    pitch_norm = pitch_candidate["predicted_norm"]

    if yaw_candidate["pulse_seconds"] <= 0.0 and pitch_candidate["pulse_seconds"] <= 0.0:
        # Fall back to the smaller current error axis if both candidates collapse to zero.
        if x_error >= y_error:
            axis = "yaw"
            key = "yaw_left" if marker.dx < 0 else "yaw_right"
            estimate = _estimate_pulse_seconds(
                axis=axis,
                axis_error_px=x_error,
                axis_error_radians=_axis_error_radians(marker, axis, config),
                config=config,
            )
        else:
            axis = "pitch"
            key = "pitch_up" if marker.dy < 0 else "pitch_down"
            estimate = _estimate_pulse_seconds(
                axis=axis,
                axis_error_px=y_error,
                axis_error_radians=_axis_error_radians(marker, axis, config),
                config=config,
            )
        estimate = _tune_filled_pulse(marker, estimate, config)
        return axis, key, estimate

    use_yaw = False
    if yaw_candidate["pulse_seconds"] > 0.0 and pitch_candidate["pulse_seconds"] > 0.0:
        yaw_improvement = current_norm - yaw_norm
        pitch_improvement = current_norm - pitch_norm
        use_yaw = yaw_improvement >= pitch_improvement
    elif yaw_candidate["pulse_seconds"] > 0.0:
        use_yaw = True

    if use_yaw:
        axis = "yaw"
        key = "yaw_left" if marker.dx < 0 else "yaw_right"
        estimate = yaw_candidate["estimate"]
    else:
        axis = "pitch"
        key = "pitch_up" if marker.dy < 0 else "pitch_down"
        estimate = pitch_candidate["estimate"]

    return axis, key, estimate


def _solve_axis_candidate(
    axis: str,
    primary_error_px: float,
    cross_error_px: float,
    axis_error_radians: float,
    axis_rate_deg_per_sec: float,
    response_model: _AxisResponseModel,
    min_seconds: float,
    max_seconds: float,
    target_axis_px: float,
    pulse_mode: str,
    config: AlignConfig,
) -> dict[str, Any]:
    primary_delta_coeff = max(response_model.primary_coeff, 1e-6)
    cross_delta_coeff = max(response_model.cross_coeff, 0.0)
    if pulse_mode == "filled_ultra_fine_model":
        cross_delta_coeff *= config.ultra_fine_cross_coeff_scale
    target_primary_delta = max(0.0, primary_error_px - target_axis_px)
    effective_min_seconds = max(min_seconds, response_model.minimum_effective_seconds)
    effective_max_seconds = max_seconds
    if pulse_mode == "filled_fine_model" and response_model.samples < config.fine_model_min_samples_for_full_pulse:
        cautious_cap = (
            config.fine_yaw_uncalibrated_max_seconds if axis == "yaw" else config.fine_pitch_uncalibrated_max_seconds
        )
        effective_max_seconds = min(effective_max_seconds, cautious_cap)
    effective_max_seconds = max(effective_max_seconds, effective_min_seconds)

    if target_primary_delta <= 0.0:
        pulse_seconds = 0.0
    else:
        pulse_squared = target_primary_delta / primary_delta_coeff
        if cross_delta_coeff > 0.0 and cross_error_px < config.ultra_fine_cross_guard_px:
            allowed_cross_growth = max(0.0, config.ultra_fine_cross_guard_px - cross_error_px)
            if allowed_cross_growth > 0.0:
                pulse_squared = min(pulse_squared, allowed_cross_growth / cross_delta_coeff)
        pulse_seconds = math.sqrt(pulse_squared)
        pulse_seconds = min(effective_max_seconds, max(effective_min_seconds, pulse_seconds))

    primary_delta = primary_delta_coeff * (pulse_seconds**2)
    cross_delta = cross_delta_coeff * (pulse_seconds**2)
    predicted_primary = max(0.0, primary_error_px - primary_delta)
    predicted_cross = cross_error_px + cross_delta
    predicted_norm = math.hypot(
        max(0.0, predicted_primary - config.axis_alignment_tolerance_px),
        max(0.0, predicted_cross - config.axis_alignment_tolerance_px),
    )
    ramp_angle_degrees = 0.5 * axis_rate_deg_per_sec * max(config.response_ramp_seconds, 0.0)
    estimate = PulseEstimate(
        axis=axis,
        axis_error_px=primary_error_px,
        axis_error_radians=axis_error_radians,
        axis_error_degrees=math.degrees(axis_error_radians),
        target_hold_seconds=pulse_seconds,
        axis_rate_deg_per_sec=axis_rate_deg_per_sec,
        response_ramp_seconds=config.response_ramp_seconds,
        ramp_angle_degrees=ramp_angle_degrees,
        adjustment_scale=1.0,
        unclamped_seconds=pulse_seconds,
        pulse_seconds=pulse_seconds,
        pulse_mode=pulse_mode,
        damping_factor=1.0,
        response_primary_coeff=primary_delta_coeff,
        response_cross_coeff=cross_delta_coeff,
        target_axis_px=target_axis_px,
    )
    return {
        "pulse_seconds": pulse_seconds,
        "predicted_norm": predicted_norm,
        "estimate": estimate,
    }


def _tune_filled_pulse(marker: CompassMarker, pulse_estimate: PulseEstimate, config: AlignConfig) -> PulseEstimate:
    if not marker.is_filled:
        return pulse_estimate
    if pulse_estimate.pulse_mode != "angle":
        return pulse_estimate

    axis = pulse_estimate.axis
    axis_error_px = pulse_estimate.axis_error_px
    tuned_seconds = pulse_estimate.pulse_seconds
    pulse_mode = pulse_estimate.pulse_mode

    if axis == "pitch":
        soft_threshold = config.filled_soft_pitch_error_px
        micro_max = config.filled_micro_pitch_max_seconds
        soft_max = config.filled_soft_pitch_max_seconds
    else:
        soft_threshold = config.filled_soft_yaw_error_px
        micro_max = config.filled_micro_yaw_max_seconds
        soft_max = config.filled_soft_yaw_max_seconds

    if axis_error_px <= config.filled_micro_error_px:
        tuned_seconds = min(tuned_seconds * config.filled_micro_scale, micro_max)
        pulse_mode = "filled_micro"
    elif axis_error_px <= soft_threshold:
        tuned_seconds = min(tuned_seconds * config.filled_soft_scale, soft_max)
        pulse_mode = "filled_soft"

    return PulseEstimate(
        axis=pulse_estimate.axis,
        axis_error_px=pulse_estimate.axis_error_px,
        axis_error_radians=pulse_estimate.axis_error_radians,
        axis_error_degrees=pulse_estimate.axis_error_degrees,
        target_hold_seconds=pulse_estimate.target_hold_seconds,
        axis_rate_deg_per_sec=pulse_estimate.axis_rate_deg_per_sec,
        response_ramp_seconds=pulse_estimate.response_ramp_seconds,
        ramp_angle_degrees=pulse_estimate.ramp_angle_degrees,
        adjustment_scale=pulse_estimate.adjustment_scale,
        unclamped_seconds=pulse_estimate.unclamped_seconds,
        pulse_seconds=tuned_seconds,
        pulse_mode=pulse_mode,
        damping_factor=pulse_estimate.damping_factor,
        response_primary_coeff=pulse_estimate.response_primary_coeff,
        response_cross_coeff=pulse_estimate.response_cross_coeff,
        target_axis_px=pulse_estimate.target_axis_px,
    )


def _apply_overshoot_damping(
    marker: CompassMarker,
    pulse_estimate: PulseEstimate,
    last_command: _PreviousCommand | None,
    config: AlignConfig,
) -> PulseEstimate:
    if last_command is None or last_command.axis != pulse_estimate.axis:
        return pulse_estimate

    current_error = _signed_axis_error(marker, pulse_estimate.axis)
    sign_flipped = last_command.signed_axis_error * current_error < 0.0
    catastrophic = _is_catastrophic_overshoot(marker, last_command, config)
    previous_magnitude = abs(last_command.signed_axis_error)
    current_magnitude = abs(current_error)
    non_improving = (
        "filled_" in pulse_estimate.pulse_mode
        and current_magnitude >= (previous_magnitude - config.fine_model_observation_floor_px)
    )
    if not sign_flipped and not catastrophic and not non_improving:
        return pulse_estimate

    if catastrophic:
        damping_factor = config.catastrophic_overshoot_damping_factor
        damped_seconds = min(
            pulse_estimate.pulse_seconds * damping_factor,
            config.catastrophic_overshoot_max_pulse_seconds,
        )
        pulse_mode_suffix = "catastrophic_overshoot_damped"
    elif non_improving and not sign_flipped:
        damping_factor = config.overshoot_damping_factor
        damped_seconds = pulse_estimate.pulse_seconds * damping_factor
        pulse_mode_suffix = "non_improving_damped"
    else:
        damping_factor = config.overshoot_damping_factor
        damped_seconds = pulse_estimate.pulse_seconds * damping_factor
        pulse_mode_suffix = "overshoot_damped"

    return PulseEstimate(
        axis=pulse_estimate.axis,
        axis_error_px=pulse_estimate.axis_error_px,
        axis_error_radians=pulse_estimate.axis_error_radians,
        axis_error_degrees=pulse_estimate.axis_error_degrees,
        target_hold_seconds=pulse_estimate.target_hold_seconds,
        axis_rate_deg_per_sec=pulse_estimate.axis_rate_deg_per_sec,
        response_ramp_seconds=pulse_estimate.response_ramp_seconds,
        ramp_angle_degrees=pulse_estimate.ramp_angle_degrees,
        adjustment_scale=pulse_estimate.adjustment_scale,
        unclamped_seconds=pulse_estimate.unclamped_seconds,
        pulse_seconds=damped_seconds,
        pulse_mode=f"{pulse_estimate.pulse_mode}_{pulse_mode_suffix}",
        damping_factor=damping_factor,
        response_primary_coeff=pulse_estimate.response_primary_coeff,
        response_cross_coeff=pulse_estimate.response_cross_coeff,
        target_axis_px=pulse_estimate.target_axis_px,
    )


def _axis_error_radians(marker: CompassMarker, axis: str, config: AlignConfig) -> float:
    radius = max(config.compass_radius_px, 1.0)
    if axis == "yaw":
        component_px = abs(marker.dx)
    else:
        component_px = abs(marker.dy)

    component = min(1.0, component_px / radius)
    radial = min(1.0, marker.normalized_radius)
    z_abs = math.sqrt(max(0.0, 1.0 - (radial * radial)))

    if marker.is_hollow:
        if component <= 1e-6:
            result = math.pi / 2.0
        else:
            result = math.atan2(z_abs, component)
        if marker.phase_adjustment_radians <= math.radians(config.hollow_edge_phase_epsilon_degrees):
            result = max(result, math.radians(config.hollow_edge_push_degrees))
        return result

    return math.atan2(component, max(z_abs, 1e-6))


def _estimate_pulse_seconds(
    axis: str,
    axis_error_px: float,
    axis_error_radians: float,
    config: AlignConfig,
) -> PulseEstimate:
    axis_error_degrees = math.degrees(axis_error_radians)
    if axis == "yaw":
        axis_rate_deg_per_sec = config.yaw_rate_deg_per_sec
        scale = config.yaw_adjustment_scale
    else:
        axis_rate_deg_per_sec = config.pitch_rate_deg_per_sec
        scale = config.pitch_adjustment_scale

    target_hold_seconds = _solve_hold_seconds_for_angle(
        target_angle_degrees=axis_error_degrees,
        axis_rate_deg_per_sec=axis_rate_deg_per_sec,
        response_ramp_seconds=config.response_ramp_seconds,
    )
    unclamped_seconds = target_hold_seconds * scale
    pulse_seconds = min(config.max_pulse_seconds, unclamped_seconds)
    ramp_angle_degrees = 0.5 * axis_rate_deg_per_sec * max(config.response_ramp_seconds, 0.0)
    return PulseEstimate(
        axis=axis,
        axis_error_px=axis_error_px,
        axis_error_radians=axis_error_radians,
        axis_error_degrees=axis_error_degrees,
        target_hold_seconds=target_hold_seconds,
        axis_rate_deg_per_sec=axis_rate_deg_per_sec,
        response_ramp_seconds=config.response_ramp_seconds,
        ramp_angle_degrees=ramp_angle_degrees,
        adjustment_scale=scale,
        unclamped_seconds=unclamped_seconds,
        pulse_seconds=pulse_seconds,
    )


def _solve_hold_seconds_for_angle(
    target_angle_degrees: float,
    axis_rate_deg_per_sec: float,
    response_ramp_seconds: float,
) -> float:
    if target_angle_degrees <= 0.0:
        return 0.0

    if axis_rate_deg_per_sec <= 0.0:
        return 0.0

    if response_ramp_seconds <= 0.0:
        return target_angle_degrees / axis_rate_deg_per_sec

    ramp_angle_degrees = 0.5 * axis_rate_deg_per_sec * response_ramp_seconds
    if target_angle_degrees <= ramp_angle_degrees:
        acceleration_deg_per_sec2 = axis_rate_deg_per_sec / response_ramp_seconds
        return math.sqrt((2.0 * target_angle_degrees) / acceleration_deg_per_sec2)

    cruise_angle_degrees = target_angle_degrees - ramp_angle_degrees
    return response_ramp_seconds + (cruise_angle_degrees / axis_rate_deg_per_sec)


def _save_debug_snapshot(context: Context, name: str, image: Any) -> Path:
    if context.vision is not None:
        return context.vision.save_debug_snapshot(name, image)

    context.debug_snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = context.debug_snapshot_dir / f"{timestamp}_{name}.png"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write debug snapshot: {path}")
    return path


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
            compass_radius_px=STANDALONE_COMPASS_RADIUS_PX,
            compass_control_rotation_degrees=STANDALONE_COMPASS_CONTROL_ROTATION_DEGREES,
            settle_seconds_after_input=(
                STANDALONE_SETTLE_SECONDS_AFTER_INPUT
                if STANDALONE_SETTLE_SECONDS_AFTER_INPUT is not None
                else 2.0
            ),
            idle_read_backoff_seconds=(
                STANDALONE_IDLE_READ_BACKOFF_SECONDS
                if STANDALONE_IDLE_READ_BACKOFF_SECONDS is not None
                else 0.25
            ),
            timeout_seconds=STANDALONE_TIMEOUT_SECONDS,
            pitch_adjustment_scale=STANDALONE_PITCH_ADJUSTMENT_SCALE,
            yaw_adjustment_scale=STANDALONE_YAW_ADJUSTMENT_SCALE,
            max_pulse_seconds=STANDALONE_MAX_PULSE_SECONDS,
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
