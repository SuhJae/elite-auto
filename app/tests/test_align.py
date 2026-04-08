from __future__ import annotations

import logging
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.actions.align import (
    AlignConfig,
    AlignToTargetCompass,
    CompassMarker,
    CompassReadResult,
    _AxisResponseModel,
    _PreviousCommand,
    _apply_overshoot_damping,
    _build_response_models,
    _estimate_pulse_seconds,
    _select_alignment_command,
    _settle_after_input_seconds,
    _tune_filled_pulse,
    _update_response_models,
    detect_compass_marker,
)
from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import ShipState


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class FakeStateReader:
    def __init__(self, state: ShipState) -> None:
        self._state = state

    def snapshot(self) -> ShipState:
        return self._state

    def is_docked(self) -> bool:
        return self._state.is_docked

    def is_mass_locked(self) -> bool:
        return self._state.is_mass_locked

    def is_supercruise(self) -> bool:
        return self._state.is_supercruise

    def cargo_count(self) -> int:
        return self._state.cargo_count

    def gui_focus(self) -> int | None:
        return self._state.gui_focus


class FakeEventStream:
    def poll_events(self, limit: int | None = None) -> list[object]:
        return []


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], timeline: list[tuple[str, object]]) -> None:
        self._frames = frames
        self._timeline = timeline
        self._index = 0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def grab(self, region=None):  # noqa: ANN001 - test double
        self._timeline.append(("grab", region))
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        return frame.copy()


class FlakyCapture:
    def __init__(self, outcomes: list[object], timeline: list[tuple[str, object]]) -> None:
        self._outcomes = outcomes
        self._timeline = timeline
        self._index = 0

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def grab(self, region=None):  # noqa: ANN001 - test double
        self._timeline.append(("grab", region))
        outcome = self._outcomes[min(self._index, len(self._outcomes) - 1)]
        self._index += 1
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return None
        return outcome.copy()


class FakeInputAdapter:
    def __init__(self, timeline: list[tuple[str, object]]) -> None:
        self._timeline = timeline
        self.holds: list[tuple[str, float]] = []

    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        self._timeline.append(("press", (key, presses, interval)))

    def key_down(self, key: str) -> None:
        self._timeline.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self._timeline.append(("key_up", key))

    def hold(self, key: str, seconds: float) -> None:
        self.holds.append((key, seconds))
        self._timeline.append(("hold", (key, seconds)))


class FakeVision:
    def __init__(self, root: Path) -> None:
        self._root = root
        self.saved_paths: list[Path] = []

    def match_template(self, image, template, region=None, threshold: float = 0.9):  # noqa: ANN001
        return None

    def save_debug_snapshot(self, name: str, image) -> Path:  # noqa: ANN001
        path = self._root / f"{name}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write debug snapshot: {path}")
        self.saved_paths.append(path)
        return path


def build_state(*, is_docked: bool) -> ShipState:
    return ShipState(
        is_docked=is_docked,
        is_mass_locked=False,
        is_supercruise=False,
        cargo_count=0,
        gui_focus=None,
        status_flags=0,
    )


def build_context(
    frames: list[np.ndarray],
    timeline: list[tuple[str, object]],
    debug_snapshot_dir: Path,
) -> tuple[Context, FakeInputAdapter, FakeVision]:
    logger = logging.getLogger("elite_auto.test_align")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    config = AppConfig.default()
    input_adapter = FakeInputAdapter(timeline)
    vision = FakeVision(debug_snapshot_dir)
    context = Context(
        config=config,
        logger=logger,
        debug_snapshot_dir=debug_snapshot_dir,
        state_reader=FakeStateReader(build_state(is_docked=False)),
        event_stream=FakeEventStream(),
        input_adapter=input_adapter,
        ship_control=None,
        capture=FakeCapture(frames, timeline),
        vision=vision,
    )
    return context, input_adapter, vision


def load_fixture(name: str) -> np.ndarray:
    path = FIXTURES_DIR / name
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Fixture could not be loaded: {path}")
    return image


def make_marker(state: str, dx: float, dy: float, config: AlignConfig) -> CompassReadResult:
    marker_x = config.center_x + dx
    marker_y = config.center_y + dy
    control_dx, control_dy = _rotate_offset(dx, dy, config.compass_control_rotation_radians)
    distance = float((dx**2 + dy**2) ** 0.5)
    normalized_radius = min(1.0, distance / max(config.compass_radius_px, 1.0))
    front_semisphere_radians = float(np.arcsin(normalized_radius))
    if state == "filled":
        target_off_boresight_radians = front_semisphere_radians
        phase_adjustment_radians = front_semisphere_radians
    else:
        target_off_boresight_radians = float(np.pi - front_semisphere_radians)
        phase_adjustment_radians = float(max(0.0, (np.pi / 2.0) - front_semisphere_radians))
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
        component_area=20,
        inner_occupancy=1.0 if state == "filled" else 0.1,
        outer_ring_occupancy=0.2,
        roi_region=config.roi_region(),
    )
    return CompassReadResult(status=state, marker=marker)


def _rotate_offset(dx: float, dy: float, rotation_radians: float) -> tuple[float, float]:
    cosine = float(np.cos(rotation_radians))
    sine = float(np.sin(rotation_radians))
    return (cosine * dx) - (sine * dy), (sine * dx) + (cosine * dy)


class TestCompassDetector(unittest.TestCase):
    def test_detect_compass_marker_hollow_off_center_fixture(self) -> None:
        result = detect_compass_marker(load_fixture("compass_hollow_off_center.png"), AlignConfig())

        self.assertEqual(result.status, "hollow")
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.marker_x, 734.615, delta=3.0)
        self.assertAlmostEqual(result.marker.marker_y, 824.590, delta=3.0)
        self.assertFalse(result.marker.is_aligned(AlignConfig()))

    def test_detect_compass_marker_hollow_edge_rim_fixture(self) -> None:
        result = detect_compass_marker(load_fixture("compass_hollow_edge_rim.png"), AlignConfig())

        self.assertEqual(result.status, "hollow")
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.marker_x, 711.0, delta=3.0)
        self.assertAlmostEqual(result.marker.marker_y, 798.154, delta=3.0)
        self.assertFalse(result.marker.is_aligned(AlignConfig()))

    def test_detect_compass_marker_filled_off_center_fixture(self) -> None:
        result = detect_compass_marker(load_fixture("compass_filled_off_center.png"), AlignConfig())

        self.assertEqual(result.status, "filled")
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.marker_x, 726.702, delta=3.0)
        self.assertAlmostEqual(result.marker.marker_y, 798.936, delta=3.0)
        self.assertFalse(result.marker.is_aligned(AlignConfig()))

    def test_detect_compass_marker_filled_centered_fixture(self) -> None:
        config = AlignConfig()
        result = detect_compass_marker(load_fixture("compass_filled_centered.png"), config)

        self.assertEqual(result.status, "filled")
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.marker_x, 730.578, delta=3.0)
        self.assertAlmostEqual(result.marker.marker_y, 817.844, delta=3.0)
        self.assertTrue(result.marker.is_aligned(config))

    def test_alignment_requires_strictly_under_one_pixel_per_axis(self) -> None:
        config = AlignConfig(alignment_tolerance_px=2.0, axis_alignment_tolerance_px=1.0)
        aligned_marker = make_marker("filled", dx=0.8, dy=0.9, config=config).marker
        boundary_marker = make_marker("filled", dx=0.75, dy=1.0, config=config).marker
        not_aligned_marker = make_marker("filled", dx=0.6, dy=1.2, config=config).marker

        assert aligned_marker is not None
        assert boundary_marker is not None
        assert not_aligned_marker is not None
        self.assertTrue(aligned_marker.is_aligned(config))
        self.assertFalse(boundary_marker.is_aligned(config))
        self.assertFalse(not_aligned_marker.is_aligned(config))

    def test_confirmation_alignment_uses_relaxed_deadband(self) -> None:
        config = AlignConfig(
            alignment_tolerance_px=2.0,
            axis_alignment_tolerance_px=1.0,
            confirmation_axis_tolerance_px=5.0,
            confirmation_distance_tolerance_px=6.0,
        )
        retained_marker = make_marker("filled", dx=-0.5, dy=-5.0, config=config).marker
        too_far_marker = make_marker("filled", dx=-0.5, dy=-6.0, config=config).marker

        assert retained_marker is not None
        assert too_far_marker is not None
        self.assertFalse(retained_marker.is_aligned(config))
        self.assertTrue(retained_marker.is_confirmation_aligned(config))
        self.assertFalse(too_far_marker.is_confirmation_aligned(config))


class TestAlignAction(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AlignConfig()
        self.window_patch = patch("app.actions.align._find_window_client_region", return_value=(0, 0, 1920, 1080))
        self.output_patch = patch("app.actions.align._resolve_output_region", return_value=(0, (0, 0, 1920, 1080)))
        self.window_patch.start()
        self.output_patch.start()
        self.addCleanup(self.window_patch.stop)
        self.addCleanup(self.output_patch.stop)

    def test_align_filled_left_emits_yaw_left_hold_only(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("filled", dx=-12.0, dy=-1.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToTargetCompass(config=self.config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(len(input_adapter.holds), 1)
        self.assertEqual(input_adapter.holds[0][0], context.config.controls.yaw_left)
        self.assertGreater(input_adapter.holds[0][1], 0.0)

    def test_align_filled_below_emits_pitch_down_hold_only(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("filled", dx=1.0, dy=12.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToTargetCompass(config=self.config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(len(input_adapter.holds), 1)
        self.assertEqual(input_adapter.holds[0][0], context.config.controls.pitch_down)

    def test_align_hollow_centered_emits_pitch_up_hold_only(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("hollow", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToTargetCompass(config=self.config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(len(input_adapter.holds), 1)
        self.assertEqual(input_adapter.holds[0][0], context.config.controls.pitch_up)

    def test_hollow_edge_never_produces_zero_hold(self) -> None:
        marker = make_marker("hollow", dx=-32.0, dy=0.0, config=self.config).marker
        assert marker is not None

        _, _, estimate = _select_alignment_command(marker, self.config)

        self.assertGreater(estimate.pulse_seconds, 0.0)

    def test_align_upper_edge_uses_pitch_even_with_rotation_config(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 4
        rotated_config = AlignConfig(compass_control_rotation_degrees=-90.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("filled", dx=0.0, dy=-12.0, config=rotated_config),
                make_marker("filled", dx=0.0, dy=0.0, config=rotated_config),
                make_marker("filled", dx=0.0, dy=0.0, config=rotated_config),
                make_marker("filled", dx=0.0, dy=0.0, config=rotated_config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToTargetCompass(config=rotated_config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(len(input_adapter.holds), 1)
        self.assertEqual(input_adapter.holds[0][0], context.config.controls.pitch_up)

    def test_align_waits_for_settle_after_input_before_next_capture(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            context, _, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("filled", dx=-12.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToTargetCompass(config=self.config).run(context)

        self.assertTrue(result.success)
        self.assertGreaterEqual(len(timeline), 4)
        self.assertEqual(timeline[0][0], "grab")
        self.assertEqual(timeline[1][0], "hold")
        hold_seconds = next(value[1] for kind, value in timeline if kind == "hold")
        expected_settle = _settle_after_input_seconds(hold_seconds, self.config)
        self.assertEqual(timeline[2], ("sleep", expected_settle))
        self.assertEqual(timeline[3][0], "grab")

    def test_align_succeeds_after_three_settled_centered_reads(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 3

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToTargetCompass(config=self.config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(input_adapter.holds, [])

    def test_align_keeps_confirming_when_marker_drifts_within_confirmation_deadband(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 3

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("filled", dx=-0.37, dy=0.109, config=self.config),
                make_marker("filled", dx=-0.5, dy=-5.0, config=self.config),
                make_marker("filled", dx=-0.3, dy=-4.6, config=self.config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToTargetCompass(config=self.config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(input_adapter.holds, [])

    def test_align_repeated_ambiguous_reads_fail_with_debug_snapshot(self) -> None:
        timeline: list[tuple[str, object]] = []
        blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frames = [blank_frame, blank_frame]

        with tempfile.TemporaryDirectory() as temp_dir:
            context, _, vision = build_context(frames, timeline, Path(temp_dir))
            reads = [CompassReadResult(status="ambiguous"), CompassReadResult(status="ambiguous")]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToTargetCompass(config=self.config).run(context)

            self.assertEqual(len(vision.saved_paths), 1)
            self.assertTrue(vision.saved_paths[0].exists())

        self.assertFalse(result.success)
        assert result.debug is not None
        self.assertIn("debug_snapshot", result.debug)

    def test_ramp_aware_yaw_estimate_is_not_tiny(self) -> None:
        estimate = _estimate_pulse_seconds(
            axis="yaw",
            axis_error_px=12.978,
            axis_error_radians=math.radians(24.017),
            config=self.config,
        )

        self.assertGreater(estimate.target_hold_seconds, 2.0)
        self.assertGreater(estimate.pulse_seconds, 2.0)
        self.assertEqual(estimate.axis_rate_deg_per_sec, self.config.yaw_rate_deg_per_sec)

    def test_small_filled_pitch_error_uses_fine_model_short_pulse(self) -> None:
        marker = make_marker("filled", dx=0.0, dy=-5.083, config=self.config).marker
        assert marker is not None

        _, _, estimate = _select_alignment_command(marker, self.config)

        self.assertEqual(estimate.axis, "pitch")
        self.assertEqual(estimate.pulse_mode, "filled_fine_model")
        self.assertLessEqual(estimate.pulse_seconds, self.config.fine_pitch_max_seconds)

    def test_medium_filled_yaw_error_switches_to_fine_model_earlier(self) -> None:
        marker = make_marker("filled", dx=-13.38, dy=0.0, config=self.config).marker
        assert marker is not None

        _, _, estimate = _select_alignment_command(marker, self.config)

        self.assertEqual(estimate.axis, "yaw")
        self.assertEqual(estimate.pulse_mode, "filled_fine_model")
        self.assertLessEqual(estimate.pulse_seconds, self.config.fine_yaw_uncalibrated_max_seconds)

    def test_medium_filled_pitch_error_switches_to_fine_model_earlier(self) -> None:
        marker = make_marker("filled", dx=-5.94, dy=-12.3, config=self.config).marker
        assert marker is not None

        _, key_name, estimate = _select_alignment_command(marker, self.config)

        self.assertEqual(key_name, "pitch_up")
        self.assertEqual(estimate.pulse_mode, "filled_fine_model")
        self.assertLessEqual(estimate.pulse_seconds, self.config.fine_pitch_uncalibrated_max_seconds)

    def test_filled_fine_model_uses_short_pitch_correction_for_small_x_positive_small_y_negative(self) -> None:
        marker = make_marker("filled", dx=2.13, dy=-4.389, config=self.config).marker
        assert marker is not None

        _, key_name, estimate = _select_alignment_command(marker, self.config)

        self.assertEqual(key_name, "pitch_up")
        self.assertEqual(estimate.pulse_mode, "filled_fine_model")
        self.assertLessEqual(estimate.pulse_seconds, self.config.fine_pitch_max_seconds)

    def test_ultra_fine_model_targets_one_pixel_band(self) -> None:
        marker = make_marker("filled", dx=-0.7, dy=-2.2, config=self.config).marker
        assert marker is not None

        _, key_name, estimate = _select_alignment_command(marker, self.config)

        self.assertEqual(key_name, "pitch_up")
        self.assertEqual(estimate.pulse_mode, "filled_ultra_fine_model")
        self.assertEqual(estimate.target_axis_px, self.config.ultra_fine_target_axis_px)
        self.assertLessEqual(estimate.pulse_seconds, self.config.ultra_fine_pitch_max_seconds)

    def test_settle_after_input_adds_damping_time(self) -> None:
        settle = _settle_after_input_seconds(0.35, self.config)
        self.assertEqual(settle, self.config.settle_seconds_after_input)

        long_settle = _settle_after_input_seconds(1.6, self.config)
        self.assertEqual(
            long_settle,
            self.config.settle_seconds_after_input
            + min(
                1.6 - self.config.response_ramp_seconds,
                self.config.post_input_extra_settle_seconds_max,
            ),
        )

    def test_response_model_updates_from_real_pulse_feedback(self) -> None:
        models = _build_response_models(self.config)
        marker = make_marker("filled", dx=0.0, dy=-3.2, config=self.config).marker
        assert marker is not None
        update = _update_response_models(
            marker,
            last_command=_PreviousCommand(
                axis="pitch",
                signed_axis_error=-4.6,
                pulse_seconds=0.2,
                marker_dx=0.0,
                marker_dy=-4.6,
                pulse_mode="filled_ultra_fine_model",
            ),
            response_models=models,
            config=self.config,
        )

        assert update is not None
        self.assertIn("primary_coeff_after", update)
        self.assertGreater(models["pitch"].samples, 0)
        self.assertNotEqual(
            models["pitch"].primary_coeff,
            self.config.fine_pitch_primary_delta_coeff,
        )

    def test_response_model_ignores_pathological_cross_growth(self) -> None:
        models = _build_response_models(self.config)
        marker = make_marker("filled", dx=-4.22, dy=-7.06, config=self.config).marker
        assert marker is not None

        update = _update_response_models(
            marker,
            last_command=_PreviousCommand(
                axis="pitch",
                signed_axis_error=-7.333,
                pulse_seconds=0.1,
                marker_dx=-2.093,
                marker_dy=-7.333,
                pulse_mode="filled_fine_model",
            ),
            response_models=models,
            config=self.config,
        )

        assert update is not None
        self.assertEqual(models["pitch"].cross_coeff, self.config.fine_pitch_cross_delta_coeff)

    def test_response_model_raises_minimum_effective_seconds_after_noop(self) -> None:
        models = _build_response_models(self.config)
        marker = make_marker("filled", dx=-1.6, dy=0.0, config=self.config).marker
        assert marker is not None

        update = _update_response_models(
            marker,
            last_command=_PreviousCommand(
                axis="yaw",
                signed_axis_error=-1.6,
                pulse_seconds=0.2,
                marker_dx=-1.6,
                marker_dy=0.0,
                pulse_mode="filled_ultra_fine_model",
            ),
            response_models=models,
            config=self.config,
        )

        assert update is not None
        self.assertGreater(models["yaw"].minimum_effective_seconds, 0.0)

    def test_response_model_does_not_raise_minimum_effective_seconds_after_overshoot(self) -> None:
        models = _build_response_models(self.config)
        models["pitch"].minimum_effective_seconds = 0.0437
        marker = make_marker("filled", dx=4.538, dy=-35.85, config=self.config).marker
        assert marker is not None

        update = _update_response_models(
            marker,
            last_command=_PreviousCommand(
                axis="pitch",
                signed_axis_error=-7.06,
                pulse_seconds=0.5,
                marker_dx=4.22,
                marker_dy=-7.06,
                pulse_mode="filled_fine_model",
            ),
            response_models=models,
            config=self.config,
        )

        assert update is not None
        self.assertEqual(models["pitch"].minimum_effective_seconds, 0.0437)
        self.assertEqual(models["pitch"].samples, 0)

    def test_overshoot_damping_halves_followup_same_axis_pulse(self) -> None:
        marker = make_marker("filled", dx=0.0, dy=7.681, config=self.config).marker
        assert marker is not None
        baseline = _tune_filled_pulse(
            marker,
            _estimate_pulse_seconds(
                axis="pitch",
                axis_error_px=abs(marker.dy),
                axis_error_radians=math.radians(13.905),
                config=self.config,
            ),
            self.config,
        )

        damped = _apply_overshoot_damping(
            marker,
            baseline,
            last_command=_PreviousCommand(
                axis="pitch",
                signed_axis_error=-7.809,
                pulse_seconds=1.1767,
                marker_dx=0.0,
                marker_dy=-7.809,
                pulse_mode="filled_micro",
            ),
            config=self.config,
        )

        self.assertEqual(damped.pulse_mode, "filled_micro_overshoot_damped")
        self.assertAlmostEqual(
            damped.pulse_seconds,
            baseline.pulse_seconds * self.config.overshoot_damping_factor,
            delta=1e-6,
        )

    def test_catastrophic_overshoot_caps_followup_same_axis_pulse(self) -> None:
        marker = make_marker("filled", dx=4.538, dy=-35.85, config=self.config).marker
        assert marker is not None
        _, _, baseline = _select_alignment_command(marker, self.config)

        damped = _apply_overshoot_damping(
            marker,
            baseline,
            last_command=_PreviousCommand(
                axis="pitch",
                signed_axis_error=-7.06,
                pulse_seconds=0.5,
                marker_dx=4.22,
                marker_dy=-7.06,
                pulse_mode="filled_fine_model",
            ),
            config=self.config,
        )

        self.assertEqual(damped.pulse_mode, "angle_catastrophic_overshoot_damped")
        self.assertAlmostEqual(
            damped.pulse_seconds,
            min(
                baseline.pulse_seconds * self.config.catastrophic_overshoot_damping_factor,
                self.config.catastrophic_overshoot_max_pulse_seconds,
            ),
            delta=1e-6,
        )

    def test_non_improving_fine_model_pulse_is_damped(self) -> None:
        marker = make_marker("filled", dx=-1.375, dy=-2.021, config=self.config).marker
        assert marker is not None

        baseline = _select_alignment_command(
            marker,
            self.config,
            last_command=None,
            response_models={
                "pitch": _AxisResponseModel(
                    axis="pitch",
                    primary_coeff=20.0601,
                    cross_coeff=8.9058,
                    minimum_effective_seconds=0.1983,
                    samples=20,
                ),
                "yaw": _AxisResponseModel(
                    axis="yaw",
                    primary_coeff=6.1995,
                    cross_coeff=3.0,
                    minimum_effective_seconds=0.0,
                    samples=2,
                ),
            },
        )[2]

        damped = _apply_overshoot_damping(
            marker,
            baseline,
            last_command=_PreviousCommand(
                axis="pitch",
                signed_axis_error=-1.851,
                pulse_seconds=0.25,
                marker_dx=-0.234,
                marker_dy=-1.851,
                pulse_mode="filled_ultra_fine_model",
            ),
            config=self.config,
        )

        self.assertEqual(damped.pulse_mode, "filled_ultra_fine_model_non_improving_damped")
        self.assertAlmostEqual(
            damped.pulse_seconds,
            baseline.pulse_seconds * self.config.overshoot_damping_factor,
            delta=1e-6,
        )

    def test_align_retries_transient_capture_miss(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context([frame, frame, frame], timeline, Path(temp_dir))
            context.capture = FlakyCapture(
                [RuntimeError("dxcam returned no frame. Check monitor/output availability."), frame, frame, frame],
                timeline,
            )
            reads = [
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToTargetCompass(config=self.config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(input_adapter.holds, [])
        self.assertIn(("sleep", self.config.capture_retry_interval_seconds), timeline)

    def test_align_backs_off_after_missing_read(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")

        with tempfile.TemporaryDirectory() as temp_dir:
            context, _, _ = build_context([frame, frame, frame, frame], timeline, Path(temp_dir))
            reads = [
                CompassReadResult(status="missing"),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
                make_marker("filled", dx=0.0, dy=0.0, config=self.config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToTargetCompass(config=self.config).run(context)

        self.assertTrue(result.success)
        self.assertGreaterEqual(len(timeline), 3)
        self.assertEqual(timeline[0][0], "grab")
        self.assertEqual(timeline[1], ("sleep", self.config.idle_read_backoff_seconds))
        self.assertEqual(timeline[2][0], "grab")


if __name__ == "__main__":
    unittest.main()
