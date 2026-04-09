from __future__ import annotations

import itertools
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.actions.align import (
    AlignConfig,
    AlignToTargetCompass,
    _AxisDynamicsProfile,
    _FinalPhaseEpisode,
    CompassMarker,
    CompassReadResult,
    _AxisControllerState,
    _SteeringState,
    _choose_final_phase_action,
    _compute_axis_controller_output,
    _create_final_phase_controller_state,
    _desired_key_for_output,
    _detect_final_reticle,
    _slew_limit,
    _update_final_phase_model_from_episode,
    detect_compass_marker,
)
from app.actions.track_center_reticle import ReticleDetection
from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import ShipState


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


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
        self.active_keys: set[str] = set()

    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        self._timeline.append(("press", (key, presses, interval)))

    def key_down(self, key: str) -> None:
        self.active_keys.add(key)
        self._timeline.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.active_keys.discard(key)
        self._timeline.append(("key_up", key))

    def hold(self, key: str, seconds: float) -> None:
        self.holds.append((key, seconds))
        self._timeline.append(("hold", (key, seconds)))


class FakeMonotonic:
    def __init__(self, start: float = 0.0, step: float = 0.05) -> None:
        self._value = start
        self._step = step

    def __call__(self) -> float:
        current = self._value
        self._value += self._step
        return current


class SequencedMonotonic:
    def __init__(self, start: float = 0.0, steps: list[float] | None = None, default_step: float = 0.05) -> None:
        self._value = start
        self._steps = steps or []
        self._default_step = default_step
        self._index = 0

    def __call__(self) -> float:
        current = self._value
        if self._index < len(self._steps):
            self._value += self._steps[self._index]
        else:
            self._value += self._default_step
        self._index += 1
        return current


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


def make_dynamic_center_frame(
    *,
    config: AlignConfig,
    center_offset_x: int,
    center_offset_y: int,
    marker_offset_x: int,
    marker_offset_y: int,
) -> np.ndarray:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    center = (config.center_x + center_offset_x, config.center_y + center_offset_y)
    ring_color = (255, 80, 0)
    outer_ring_color = (210, 70, 10)
    marker_color = (255, 255, 255)
    inner_radius = int(round(config.inner_ring_radius_px))
    cv2.circle(frame, center, inner_radius, ring_color, 2)
    outer_radius = int(round(config.outer_ring_radius_px))
    segment_span = 28
    for start in (20, 110, 200, 290):
        cv2.ellipse(frame, center, (outer_radius, outer_radius), 0, start, start + segment_span, outer_ring_color, 2)
    cv2.circle(frame, (center[0] + marker_offset_x, center[1] + marker_offset_y), 4, marker_color, -1)
    return frame


def make_center_dot_locked_frame_with_distractor(
    *,
    config: AlignConfig,
    center_offset_x: int,
    center_offset_y: int,
    marker_offset_x: int,
    marker_offset_y: int,
    distractor_offset_x: int,
    distractor_offset_y: int,
) -> np.ndarray:
    frame = make_dynamic_center_frame(
        config=config,
        center_offset_x=center_offset_x,
        center_offset_y=center_offset_y,
        marker_offset_x=marker_offset_x,
        marker_offset_y=marker_offset_y,
    )
    center = (config.center_x + center_offset_x, config.center_y + center_offset_y)
    ring_color = (255, 80, 0)
    outer_ring_color = (210, 70, 10)
    outer_radius = int(round(config.outer_ring_radius_px))
    inner_radius = int(round(config.inner_ring_radius_px))

    cv2.circle(frame, center, 2, (255, 255, 255), -1)

    distractor_center = (center[0] + distractor_offset_x, center[1] + distractor_offset_y)
    cv2.circle(frame, distractor_center, inner_radius, ring_color, 2)
    for start in (0, 90, 180, 270):
        cv2.ellipse(frame, distractor_center, (outer_radius, outer_radius), 0, start, start + 60, outer_ring_color, 2)

    return frame


def make_ring_only_frame(
    *,
    config: AlignConfig,
    center_offset_x: int,
    center_offset_y: int,
) -> np.ndarray:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    center = (config.center_x + center_offset_x, config.center_y + center_offset_y)
    ring_color = (255, 80, 0)
    outer_ring_color = (210, 70, 10)
    inner_radius = int(round(config.inner_ring_radius_px))
    outer_radius = int(round(config.outer_ring_radius_px))
    cv2.circle(frame, center, inner_radius, ring_color, 2)
    for start in (20, 110, 200, 290):
        cv2.ellipse(frame, center, (outer_radius, outer_radius), 0, start, start + 28, outer_ring_color, 2)
    return frame


def make_filled_blob_frame(
    *,
    config: AlignConfig,
    center_offset_x: int,
    center_offset_y: int,
) -> np.ndarray:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    center = (config.center_x + center_offset_x, config.center_y + center_offset_y)
    cv2.circle(frame, center, int(round(config.inner_ring_radius_px + 2)), (255, 120, 60), -1)
    return frame


def make_partial_arc_frame(
    *,
    config: AlignConfig,
    center_offset_x: int,
    center_offset_y: int,
) -> np.ndarray:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    center = (config.center_x + center_offset_x, config.center_y + center_offset_y)
    outer_radius = int(round(config.outer_ring_radius_px))
    inner_radius = int(round(config.inner_ring_radius_px))
    cv2.ellipse(frame, center, (inner_radius, inner_radius), 0, 210, 320, (255, 80, 0), 2)
    cv2.ellipse(frame, center, (outer_radius, outer_radius), 0, 210, 260, (210, 70, 10), 2)
    return frame


def make_perspective_ellipse_frame(
    *,
    config: AlignConfig,
    center_offset_x: int,
    center_offset_y: int,
    marker_offset_x: int,
    marker_offset_y: int,
    inner_axis_x: int,
    inner_axis_y: int,
    outer_axis_x: int,
    outer_axis_y: int,
    angle_degrees: float = 0.0,
) -> np.ndarray:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    center = (config.center_x + center_offset_x, config.center_y + center_offset_y)
    cv2.ellipse(frame, center, (inner_axis_x, inner_axis_y), angle_degrees, 0, 360, (255, 80, 0), 2)
    for start in (20, 110, 200, 290):
        cv2.ellipse(
            frame,
            center,
            (outer_axis_x, outer_axis_y),
            angle_degrees,
            start,
            start + 28,
            (210, 70, 10),
            2,
        )
    cv2.circle(frame, (center[0] + marker_offset_x, center[1] + marker_offset_y), 4, (255, 255, 255), -1)
    return frame


def _rotate_offset(dx: float, dy: float, rotation_radians: float) -> tuple[float, float]:
    cosine = float(np.cos(rotation_radians))
    sine = float(np.sin(rotation_radians))
    return (cosine * dx) - (sine * dy), (sine * dx) + (cosine * dy)


class TestCompassDetector(unittest.TestCase):
    def test_detect_compass_marker_hollow_off_center_fixture(self) -> None:
        result = detect_compass_marker(load_fixture("compass_hollow_off_center.png"), AlignConfig())

        self.assertEqual(result.status, "hollow")
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.marker_x, 734.615, delta=3.1)
        self.assertAlmostEqual(result.marker.marker_y, 824.590, delta=3.1)
        self.assertFalse(result.marker.is_aligned(AlignConfig()))

    def test_detect_compass_marker_hollow_edge_rim_fixture(self) -> None:
        result = detect_compass_marker(load_fixture("compass_hollow_edge_rim.png"), AlignConfig())

        self.assertEqual(result.status, "hollow")
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.marker_x, 711.0, delta=3.0)
        self.assertAlmostEqual(result.marker.marker_y, 798.154, delta=3.1)
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

    def test_detect_compass_marker_tracks_dynamic_ring_center(self) -> None:
        config = AlignConfig()
        image = make_dynamic_center_frame(
            config=config,
            center_offset_x=10,
            center_offset_y=-6,
            marker_offset_x=8,
            marker_offset_y=5,
        )

        result = detect_compass_marker(image, config)

        self.assertEqual(result.status, "filled")
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.dx, 8.0, delta=2.0)
        self.assertAlmostEqual(result.marker.dy, 5.0, delta=2.0)
        self.assertAlmostEqual(result.marker.compass_center_x or 0.0, config.center_x + 10, delta=2.0)
        self.assertAlmostEqual(result.marker.compass_center_y or 0.0, config.center_y - 6, delta=2.0)

    def test_detect_compass_marker_keeps_circle_locked_to_center_dot(self) -> None:
        config = AlignConfig()
        image = make_center_dot_locked_frame_with_distractor(
            config=config,
            center_offset_x=7,
            center_offset_y=-4,
            marker_offset_x=9,
            marker_offset_y=6,
            distractor_offset_x=22,
            distractor_offset_y=-3,
        )

        result = detect_compass_marker(image, config)

        self.assertEqual(result.status, "filled")
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.dx, 9.0, delta=2.0)
        self.assertAlmostEqual(result.marker.dy, 6.0, delta=2.0)
        self.assertAlmostEqual(result.marker.compass_center_x or 0.0, config.center_x + 7, delta=2.0)
        self.assertAlmostEqual(result.marker.compass_center_y or 0.0, config.center_y - 4, delta=2.0)

    def test_detect_compass_marker_tracks_perspective_ellipse(self) -> None:
        config = AlignConfig()
        image = make_perspective_ellipse_frame(
            config=config,
            center_offset_x=9,
            center_offset_y=-7,
            marker_offset_x=8,
            marker_offset_y=5,
            inner_axis_x=21,
            inner_axis_y=26,
            outer_axis_x=30,
            outer_axis_y=35,
            angle_degrees=12.0,
        )

        result = detect_compass_marker(image, config)

        self.assertEqual(result.status, "filled")
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.dx, 8.0, delta=3.0)
        self.assertAlmostEqual(result.marker.dy, 5.0, delta=3.0)
        self.assertAlmostEqual(result.marker.compass_center_x or 0.0, config.center_x + 9, delta=3.0)
        self.assertAlmostEqual(result.marker.compass_center_y or 0.0, config.center_y - 7, delta=3.0)

    def test_detect_compass_marker_tracks_far_offset_circle_near_camera_limit(self) -> None:
        config = AlignConfig()
        image = make_dynamic_center_frame(
            config=config,
            center_offset_x=50,
            center_offset_y=0,
            marker_offset_x=-10,
            marker_offset_y=0,
        )

        result = detect_compass_marker(image, config)

        self.assertEqual(result.status, "filled")
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.compass_center_x or 0.0, config.center_x + 50, delta=4.0)
        self.assertAlmostEqual(result.marker.compass_center_y or 0.0, config.center_y, delta=4.0)
        self.assertAlmostEqual(result.marker.dx, -10.0, delta=4.0)
        self.assertAlmostEqual(result.marker.dy, 0.0, delta=4.0)

    def test_detect_compass_marker_rejects_far_circle_from_center_dot(self) -> None:
        config = AlignConfig(center_dot_circle_center_max_distance_px=8.0)
        image = make_center_dot_locked_frame_with_distractor(
            config=config,
            center_offset_x=5,
            center_offset_y=-3,
            marker_offset_x=7,
            marker_offset_y=4,
            distractor_offset_x=36,
            distractor_offset_y=18,
        )

        result = detect_compass_marker(image, config)

        self.assertIn(result.status, {"filled", "hollow"})
        assert result.marker is not None
        self.assertAlmostEqual(result.marker.compass_center_x or 0.0, config.center_x + 5, delta=2.0)
        self.assertAlmostEqual(result.marker.compass_center_y or 0.0, config.center_y - 3, delta=2.0)

    def test_detect_compass_marker_returns_circle_only_when_ring_has_no_marker(self) -> None:
        config = AlignConfig()
        image = make_ring_only_frame(config=config, center_offset_x=9, center_offset_y=-7)

        result = detect_compass_marker(image, config)

        self.assertEqual(result.status, "circle_only")
        self.assertIsNone(result.marker)

    def test_detect_compass_marker_rejects_filled_blob_without_marker(self) -> None:
        config = AlignConfig()
        image = make_filled_blob_frame(config=config, center_offset_x=8, center_offset_y=-5)

        result = detect_compass_marker(image, config)

        self.assertIn(result.status, {"missing", "circle_only", "ambiguous"})
        self.assertIsNone(result.marker)

    def test_detect_compass_marker_rejects_partial_arc_without_marker(self) -> None:
        config = AlignConfig()
        image = make_partial_arc_frame(config=config, center_offset_x=-10, center_offset_y=6)

        result = detect_compass_marker(image, config)

        self.assertIn(result.status, {"missing", "circle_only", "ambiguous"})
        self.assertIsNone(result.marker)

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
        retained_marker = make_marker("filled", dx=-0.5, dy=-4.9, config=config).marker
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

    def test_align_diagonal_error_can_activate_pitch_and_yaw_together(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 8
        config = AlignConfig(alignment_dwell_seconds=0.15, near_center_consensus_enabled=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("filled", dx=-12.0, dy=-10.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertTrue(result.success)
        self.assertIn(("key_down", context.config.controls.yaw_left), timeline)
        self.assertIn(("key_down", context.config.controls.pitch_up), timeline)
        self.assertEqual(input_adapter.active_keys, set())

    def test_align_releases_one_axis_when_only_other_axis_still_needs_correction(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 10
        config = AlignConfig(alignment_dwell_seconds=0.15, near_center_consensus_enabled=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            context, _, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("filled", dx=12.0, dy=12.0, config=config),
                make_marker("filled", dx=12.0, dy=0.1, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.06)
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertTrue(result.success)
        self.assertLess(
            timeline.index(("key_up", context.config.controls.pitch_down)),
            timeline.index(("key_up", context.config.controls.yaw_right)),
        )

    def test_align_hollow_center_breakout_uses_pitch_up_only(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 8
        config = AlignConfig(alignment_dwell_seconds=0.15, near_center_consensus_enabled=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("hollow", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertTrue(result.success)
        self.assertIn(("key_down", context.config.controls.pitch_up), timeline)
        self.assertNotIn(("key_down", context.config.controls.yaw_left), timeline)
        self.assertNotIn(("key_down", context.config.controls.yaw_right), timeline)
        self.assertEqual(input_adapter.active_keys, set())

    def test_align_never_holds_opposite_keys_on_same_axis(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [load_fixture("compass_filled_centered.png")] * 10
        config = AlignConfig(alignment_dwell_seconds=0.15, near_center_consensus_enabled=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            context, _, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("filled", dx=-10.0, dy=0.0, config=config),
                make_marker("filled", dx=10.0, dy=0.0, config=config),
                make_marker("filled", dx=10.0, dy=0.0, config=config),
                make_marker("filled", dx=10.0, dy=0.0, config=config),
                make_marker("filled", dx=10.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.06)
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertTrue(result.success)
        self.assertLess(
            timeline.index(("key_up", context.config.controls.yaw_left)),
            timeline.index(("key_down", context.config.controls.yaw_right)),
        )

    def test_align_retries_transient_capture_miss(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(alignment_dwell_seconds=0.15, near_center_consensus_enabled=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context([frame, frame, frame], timeline, Path(temp_dir))
            context.capture = FlakyCapture(
                [RuntimeError("dxcam returned no frame. Check monitor/output availability."), frame, frame, frame],
                timeline,
            )
            reads = [
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ), patch(
                "app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(input_adapter.active_keys, set())
        self.assertIn(("sleep", self.config.capture_retry_interval_seconds), timeline)

    def test_align_oscillation_triggers_anomaly_snapshot(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        frames = [frame] * 10
        config = AlignConfig(
            alignment_dwell_seconds=0.15,
            oscillation_sign_flip_threshold=3,
            near_center_consensus_enabled=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, vision = build_context(frames, timeline, Path(temp_dir))
            reads = [
                make_marker("filled", dx=-8.0, dy=0.0, config=config),
                make_marker("filled", dx=8.0, dy=0.0, config=config),
                make_marker("filled", dx=-8.0, dy=0.0, config=config),
                make_marker("filled", dx=8.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
                make_marker("filled", dx=0.0, dy=0.0, config=config),
            ]

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.06)
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertTrue(result.success)
        self.assertTrue(any("align_oscillation" in path.name for path in vision.saved_paths))
        self.assertEqual(input_adapter.active_keys, set())

    def test_near_center_consensus_requires_pause_and_three_samples(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        frames = [frame] * 12
        config = AlignConfig(
            near_center_consensus_pause_seconds=0.10,
            near_center_consensus_samples=3,
            near_center_consensus_span_seconds=0.10,
            alignment_dwell_seconds=0.20,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context(frames, timeline, Path(temp_dir))
            reads = [make_marker("filled", dx=0.1, dy=-0.2, config=config)] * 10

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(input_adapter.active_keys, set())
        assert result.debug is not None
        self.assertIn("near_center_consensus", result.debug)
        self.assertEqual(result.debug["near_center_consensus"]["sample_count"], 3)

    def test_final_phase_does_not_succeed_from_compass_center_alone(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            near_center_consensus_enabled=False,
            final_reticle_enabled=True,
            final_reticle_stationary_seconds=0.0,
            final_reticle_required_consecutive_detections=1,
            timeout_seconds=0.6,
        )
        off_center_reticle = (
            ReticleDetection(
                found=True,
                center_x=(frame.shape[1] / 2.0) + 35.0,
                center_y=(frame.shape[0] / 2.0) - 12.0,
                outer_radius_px=41.0,
                score=2.0,
                search_region=(700, 280, 520, 520),
                metrics={"score": 2.0},
            ),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, _, _ = build_context([frame] * 10, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("filled", dx=0.0, dy=0.0, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.detect_center_reticle", return_value=off_center_reticle
            ), patch("app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertFalse(result.success)
        assert result.debug is not None
        self.assertIn("final_reticle", result.debug)
        self.assertIn("final_phase", result.debug)
        self.assertTrue(result.debug["final_reticle"]["found"])

    def test_final_phase_single_axis_choice_does_not_touch_aligned_yaw(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            near_center_consensus_enabled=False,
            final_reticle_enabled=True,
            final_reticle_stationary_seconds=0.0,
            final_reticle_required_consecutive_detections=1,
            timeout_seconds=0.6,
        )
        mostly_vertical_reticle = (
            ReticleDetection(
                found=True,
                center_x=(frame.shape[1] / 2.0) + 10.0,
                center_y=(frame.shape[0] / 2.0) + 34.0,
                outer_radius_px=41.0,
                score=2.0,
                search_region=(700, 280, 520, 520),
                metrics={"score": 2.0},
            ),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, _, _ = build_context([frame] * 10, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("filled", dx=0.0, dy=0.0, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.detect_center_reticle", return_value=mostly_vertical_reticle
            ), patch("app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertFalse(result.success)
        self.assertIn(("key_down", context.config.controls.pitch_down), timeline)
        self.assertNotIn(("key_down", context.config.controls.yaw_left), timeline)
        self.assertNotIn(("key_down", context.config.controls.yaw_right), timeline)

    def test_final_phase_chooses_larger_axis_first_when_both_are_outside_tolerance(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            near_center_consensus_enabled=False,
            final_reticle_enabled=True,
            final_reticle_stationary_seconds=0.0,
            final_reticle_required_consecutive_detections=1,
            timeout_seconds=0.6,
        )
        mostly_horizontal_reticle = (
            ReticleDetection(
                found=True,
                center_x=(frame.shape[1] / 2.0) + 46.0,
                center_y=(frame.shape[0] / 2.0) + 24.0,
                outer_radius_px=41.0,
                score=2.0,
                search_region=(700, 280, 520, 520),
                metrics={"score": 2.0},
            ),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, _, _ = build_context([frame] * 10, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("filled", dx=0.0, dy=0.0, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.detect_center_reticle", return_value=mostly_horizontal_reticle
            ), patch("app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertFalse(result.success)
        self.assertIn(("key_down", context.config.controls.yaw_right), timeline)
        self.assertNotIn(("key_down", context.config.controls.pitch_down), timeline)
        self.assertNotIn(("key_down", context.config.controls.pitch_up), timeline)

    def test_final_reticle_does_not_take_over_before_near_center_handoff(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            alignment_dwell_seconds=0.10,
            near_center_consensus_enabled=False,
            final_reticle_enabled=True,
            final_reticle_track_anywhere=False,
            final_reticle_stationary_seconds=0.0,
            timeout_seconds=0.2,
        )
        reticle_read = (
            ReticleDetection(
                found=True,
                center_x=(frame.shape[1] / 2.0) + 80.0,
                center_y=frame.shape[0] / 2.0,
                outer_radius_px=41.0,
                score=2.0,
                search_region=(700, 280, 520, 520),
                metrics={"score": 2.0},
            ),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context([frame] * 6, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("filled", dx=8.0, dy=0.0, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.detect_center_reticle", return_value=reticle_read
            ), patch("app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertFalse(result.success)
        assert result.debug is not None
        self.assertNotIn("final_reticle", result.debug)
        self.assertNotEqual(result.debug["controller_outputs"]["yaw"]["mode"], "final_reticle_nudge")
        self.assertEqual(input_adapter.active_keys, set())

    def test_final_phase_waits_full_settle_window_before_second_pulse(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            near_center_consensus_enabled=False,
            final_reticle_enabled=True,
            final_reticle_stationary_seconds=0.0,
            final_reticle_required_consecutive_detections=1,
            timeout_seconds=0.9,
        )
        off_center_read = (
            ReticleDetection(
                found=True,
                center_x=(frame.shape[1] / 2.0) + 36.0,
                center_y=frame.shape[0] / 2.0,
                outer_radius_px=41.0,
                score=2.0,
                search_region=(700, 280, 520, 520),
                metrics={"score": 2.0},
            ),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context([frame] * 10, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("filled", dx=0.0, dy=0.0, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.detect_center_reticle", return_value=off_center_read
            ), patch("app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.02)):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertFalse(result.success)
        yaw_right = context.config.controls.yaw_right
        self.assertEqual(timeline.count(("key_down", yaw_right)), 1)
        self.assertGreaterEqual(timeline.count(("key_up", yaw_right)), 1)
        self.assertEqual(input_adapter.active_keys, set())

    def test_final_phase_confirmation_requires_two_seconds_of_valid_reticle(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            near_center_consensus_enabled=False,
            final_reticle_enabled=True,
            final_reticle_stationary_seconds=0.0,
            final_reticle_required_consecutive_detections=1,
            timeout_seconds=0.7,
        )
        centered_reticle = (
            ReticleDetection(
                found=True,
                center_x=frame.shape[1] / 2.0,
                center_y=frame.shape[0] / 2.0,
                outer_radius_px=41.0,
                score=2.0,
                search_region=(700, 280, 520, 520),
                metrics={"score": 2.0},
            ),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context([frame] * 8, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("filled", dx=0.0, dy=0.0, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.detect_center_reticle", return_value=centered_reticle
            ), patch("app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.02)):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertFalse(result.success)
        assert result.debug is not None
        self.assertIn("final_phase", result.debug)
        self.assertLess(result.debug["final_phase"]["confirmation_elapsed_seconds"], config.final_phase_confirm_seconds)

    def test_hollow_small_error_forces_actuation_above_engage_threshold(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            near_center_consensus_enabled=False,
            timeout_seconds=0.2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context([frame] * 6, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("hollow", dx=5.8, dy=4.8, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertFalse(result.success)
        self.assertTrue(
            ("key_down", context.config.controls.yaw_right) in timeline
            or ("key_down", context.config.controls.pitch_down) in timeline
        )

    def test_final_reticle_missing_falls_back_to_compass_success(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            near_center_consensus_enabled=False,
            final_reticle_enabled=True,
            final_reticle_stationary_seconds=0.0,
            final_reticle_missing_compass_fallback_seconds=0.10,
            timeout_seconds=1.0,
        )
        missing_reticle = (
            ReticleDetection(
                found=False,
                search_region=(700, 280, 520, 520),
                metrics={},
            ),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context([frame] * 12, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("filled", dx=2.5, dy=-1.9, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.detect_center_reticle", return_value=missing_reticle
            ), patch("app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(input_adapter.active_keys, set())
        assert result.debug is not None
        self.assertIn("final_reticle", result.debug)
        self.assertEqual(result.debug["final_reticle"]["handoff_blocker"], "reticle_missing_fallback_to_compass")

    def test_filled_near_center_uses_pwm_pulses_before_dwell(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            near_center_consensus_enabled=False,
            timeout_seconds=0.25,
            filled_pwm_start_distance_px=16.0,
            filled_pwm_min_hold_seconds=0.03,
            filled_pwm_max_hold_seconds=0.10,
            filled_pwm_min_release_seconds=0.0,
            filled_pwm_max_release_seconds=0.06,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context([frame] * 10, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("filled", dx=4.0, dy=0.0, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertFalse(result.success)
        assert result.debug is not None
        self.assertEqual(result.debug["controller_outputs"]["yaw"]["mode"], "filled_pwm")
        yaw_right = context.config.controls.yaw_right
        self.assertIn(("key_down", yaw_right), timeline)
        self.assertIn(("key_up", yaw_right), timeline)
        self.assertEqual(input_adapter.active_keys, set())

    def test_final_reticle_handoff_waits_for_stationary_window(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            near_center_consensus_enabled=False,
            final_reticle_enabled=True,
            final_reticle_stationary_seconds=2.0,
            final_reticle_required_consecutive_detections=1,
            timeout_seconds=7.0,
        )
        centered_reticle = (
            ReticleDetection(
                found=True,
                center_x=frame.shape[1] / 2.0,
                center_y=frame.shape[0] / 2.0,
                outer_radius_px=41.0,
                score=2.0,
                search_region=(700, 280, 520, 520),
                metrics={"score": 2.0},
            ),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context([frame] * 12, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("filled", dx=1.0, dy=0.5, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.detect_center_reticle", return_value=centered_reticle
            ), patch("app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.25)):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertTrue(result.success)
        assert result.debug is not None
        self.assertIn("final_reticle", result.debug)
        self.assertTrue(result.debug["final_reticle"]["stationary_ready"])
        self.assertTrue(result.debug["final_reticle"]["engaged"])
        self.assertIn("final_phase", result.debug)
        self.assertGreaterEqual(result.debug["final_phase"]["confirmation_elapsed_seconds"], config.final_phase_confirm_seconds)

    def test_final_phase_model_update_changes_later_pulse_selection(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(final_phase_model_ema_alpha=1.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            context, _, _ = build_context([frame], timeline, Path(temp_dir))
            state = _create_final_phase_controller_state(config)

            initial_choice = _choose_final_phase_action(
                control_dx=24.0,
                control_dy=0.0,
                state=state,
                context=context,
                config=config,
            )
            episode = _FinalPhaseEpisode(
                direction="yaw_right",
                axis="yaw",
                key=context.config.controls.yaw_right,
                hold_seconds=0.09,
                commanded_error_px=24.0,
                predicted_residual_px=0.0,
                started_at=0.0,
                pulse_end_at=0.09,
                settle_until=2.09,
            )
            observed_gain = _update_final_phase_model_from_episode(
                state=state,
                episode=episode,
                control_dx=6.0,
                control_dy=0.0,
                config=config,
            )
            updated_choice = _choose_final_phase_action(
                control_dx=24.0,
                control_dy=0.0,
                state=state,
                context=context,
                config=config,
            )

        self.assertGreater(observed_gain, config.final_phase_yaw_prior_px_per_second)
        self.assertLess(updated_choice["hold_seconds"], initial_choice["hold_seconds"])

    def test_final_phase_model_update_uses_actual_hold_duration_when_available(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(final_phase_model_ema_alpha=1.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            context, _, _ = build_context([frame], timeline, Path(temp_dir))
            state = _create_final_phase_controller_state(config)

            episode = _FinalPhaseEpisode(
                direction="yaw_right",
                axis="yaw",
                key=context.config.controls.yaw_right,
                hold_seconds=0.09,
                commanded_error_px=24.0,
                predicted_residual_px=0.0,
                started_at=0.0,
                pulse_end_at=0.09,
                settle_until=2.09,
                actual_hold_seconds=0.18,
            )
            observed_gain = _update_final_phase_model_from_episode(
                state=state,
                episode=episode,
                control_dx=6.0,
                control_dy=0.0,
                config=config,
            )

        self.assertAlmostEqual(observed_gain, 100.0, places=3)

    def test_final_reticle_rejects_dashboard_false_lock_094309(self) -> None:
        frame = cv2.imread(str(REPO_ROOT / "debug_snapshots" / "20260409T094309_align_oscillation_full.png"))
        if frame is None:
            self.skipTest("Saved debug snapshot 20260409T094309_align_oscillation_full.png not available")

        debug, dx, dy = _detect_final_reticle(frame, AlignConfig(debug_window_enabled=False))

        self.assertFalse(debug["found"])
        self.assertIsNone(dx)
        self.assertIsNone(dy)

    def test_final_reticle_rejects_dashboard_false_lock_094319(self) -> None:
        frame = cv2.imread(str(REPO_ROOT / "debug_snapshots" / "20260409T094319_align_oscillation_full.png"))
        if frame is None:
            self.skipTest("Saved debug snapshot 20260409T094319_align_oscillation_full.png not available")

        debug, dx, dy = _detect_final_reticle(frame, AlignConfig(debug_window_enabled=False))

        self.assertFalse(debug["found"])
        self.assertIsNone(dx)
        self.assertIsNone(dy)

    def test_compass_circle_recovery_prefers_orb_consistent_with_marker_hint_095307(self) -> None:
        frame = cv2.imread(str(REPO_ROOT / "debug_snapshots" / "20260409T095307_align_oscillation_full.png"))
        if frame is None:
            self.skipTest("Saved debug snapshot 20260409T095307_align_oscillation_full.png not available")

        result = detect_compass_marker(frame, AlignConfig(debug_window_enabled=False))

        self.assertTrue(result.is_detected)
        self.assertIsNotNone(result.marker)
        assert result.marker is not None
        self.assertLess(result.marker.distance, 12.0)

    def test_compass_hollow_detection_tracks_small_left_orb_101035(self) -> None:
        frame = cv2.imread(str(REPO_ROOT / "debug_snapshots" / "20260409T101035_align_oscillation_full.png"))
        if frame is None:
            self.skipTest("Saved debug snapshot 20260409T101035_align_oscillation_full.png not available")

        result = detect_compass_marker(frame, AlignConfig(debug_window_enabled=False))

        self.assertTrue(result.is_detected)
        self.assertEqual(result.status, "hollow")
        assert result.marker is not None
        self.assertLess(abs(result.marker.control_dx), 5.0)
        self.assertGreater(result.marker.control_dy, 30.0)
        self.assertLess(result.marker.distance, 45.0)

    def test_compass_hollow_detection_tracks_right_shifted_orb_101049(self) -> None:
        frame = cv2.imread(str(REPO_ROOT / "debug_snapshots" / "20260409T101049_align_oscillation_full.png"))
        if frame is None:
            self.skipTest("Saved debug snapshot 20260409T101049_align_oscillation_full.png not available")

        result = detect_compass_marker(frame, AlignConfig(debug_window_enabled=False))

        self.assertTrue(result.is_detected)
        self.assertEqual(result.status, "hollow")
        assert result.marker is not None
        self.assertGreater(result.marker.control_dx, 20.0)
        self.assertLess(result.marker.control_dy, 5.0)
        self.assertLess(result.marker.distance, 40.0)

    def test_compass_hollow_detection_tracks_near_center_orb_101050(self) -> None:
        frame = cv2.imread(str(REPO_ROOT / "debug_snapshots" / "20260409T101050_align_oscillation_full.png"))
        if frame is None:
            self.skipTest("Saved debug snapshot 20260409T101050_align_oscillation_full.png not available")

        result = detect_compass_marker(frame, AlignConfig(debug_window_enabled=False))

        self.assertTrue(result.is_detected)
        self.assertEqual(result.status, "hollow")
        assert result.marker is not None
        self.assertGreater(result.marker.control_dx, 5.0)
        self.assertLess(result.marker.control_dx, 20.0)
        self.assertLess(result.marker.control_dy, 0.0)
        self.assertLess(result.marker.distance, 20.0)

    def test_compass_recovery_prefers_left_orb_over_panel_ring_200406(self) -> None:
        frame = cv2.imread(str(REPO_ROOT / "debug_snapshots" / "20260409T200406_align_oscillation_full.png"))
        if frame is None:
            self.skipTest("Saved debug snapshot 20260409T200406_align_oscillation_full.png not available")

        result = detect_compass_marker(frame, AlignConfig(debug_window_enabled=False))

        self.assertTrue(result.is_detected)
        self.assertEqual(result.status, "filled")
        assert result.marker is not None
        self.assertLess(result.marker.distance, 2.0)
        self.assertLess(result.marker.compass_center_x, 700.0)
        self.assertGreater(result.marker.compass_center_y, 850.0)

    def test_compass_recovery_prefers_left_orb_over_center_clutter_200430(self) -> None:
        frame = cv2.imread(str(REPO_ROOT / "debug_snapshots" / "20260409T200430_align_oscillation_full.png"))
        if frame is None:
            self.skipTest("Saved debug snapshot 20260409T200430_align_oscillation_full.png not available")

        result = detect_compass_marker(frame, AlignConfig(debug_window_enabled=False))

        self.assertTrue(result.is_detected)
        self.assertEqual(result.status, "filled")
        assert result.marker is not None
        self.assertLess(result.marker.distance, 3.0)
        self.assertLess(result.marker.compass_center_x, 700.0)
        self.assertLess(result.marker.compass_center_y, 820.0)

    def test_compass_recovery_prefers_upper_right_orb_over_console_lights_200514(self) -> None:
        frame = cv2.imread(str(REPO_ROOT / "debug_snapshots" / "20260409T200514_align_oscillation_full.png"))
        if frame is None:
            self.skipTest("Saved debug snapshot 20260409T200514_align_oscillation_full.png not available")

        result = detect_compass_marker(frame, AlignConfig(debug_window_enabled=False))

        self.assertTrue(result.is_detected)
        self.assertEqual(result.status, "filled")
        assert result.marker is not None
        self.assertLess(result.marker.distance, 2.0)
        self.assertGreater(result.marker.compass_center_x, 780.0)
        self.assertLess(result.marker.compass_center_y, 790.0)

    def test_final_phase_command_selection_handles_irregular_timing_inputs(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(
            near_center_consensus_enabled=False,
            final_reticle_enabled=True,
            final_reticle_stationary_seconds=0.0,
            final_reticle_required_consecutive_detections=1,
            timeout_seconds=4.5,
        )
        centered_reticle = (
            ReticleDetection(
                found=True,
                center_x=frame.shape[1] / 2.0,
                center_y=frame.shape[0] / 2.0,
                outer_radius_px=41.0,
                score=2.0,
                search_region=(700, 280, 520, 520),
                metrics={"score": 2.0},
            ),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, _ = build_context([frame] * 16, timeline, Path(temp_dir))
            reads = itertools.repeat(make_marker("filled", dx=1.0, dy=0.5, config=config))

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.detect_center_reticle", return_value=centered_reticle
            ), patch(
                "app.actions.align.time.monotonic",
                side_effect=SequencedMonotonic(steps=[0.03, 0.18, 0.07, 0.29, 0.05, 0.24, 0.06, 0.31], default_step=0.22),
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(input_adapter.active_keys, set())

    def test_runtime_dynamics_profile_scales_down_gain_for_fast_axis_response(self) -> None:
        state = _AxisControllerState(axis="yaw", previous_error=12.0, filtered_derivative=0.0, previous_output=0.0)
        profile = _AxisDynamicsProfile(axis="yaw", px_per_second=24.0, samples=3)

        output = _compute_axis_controller_output(
            axis="yaw",
            error_px=8.0,
            dt=0.2,
            marker_state="filled",
            controller_state=state,
            dynamics_profile=profile,
            config=self.config,
        )

        self.assertLess(output["gain_scale"], 1.0)
        self.assertLess(output["output"], self.config.filled_kp * 8.0)

    def test_align_releases_all_keys_on_timeout(self) -> None:
        timeline: list[tuple[str, object]] = []
        frame = load_fixture("compass_filled_centered.png")
        config = AlignConfig(timeout_seconds=0.2)

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter, vision = build_context([frame] * 10, timeline, Path(temp_dir))
            reads = [make_marker("filled", dx=-10.0, dy=0.0, config=config)] * 10

            with patch("app.actions.align.detect_compass_marker", side_effect=reads), patch(
                "app.actions.align.time.monotonic", side_effect=FakeMonotonic(step=0.05)
            ):
                result = AlignToTargetCompass(config=config).run(context)

        self.assertFalse(result.success)
        self.assertEqual(input_adapter.active_keys, set())
        self.assertTrue(any("align_timeout" in path.name for path in vision.saved_paths))

    def test_nudge_axis_commits_pulse_until_hold_window_expires(self) -> None:
        timeline: list[tuple[str, object]] = []
        input_adapter = FakeInputAdapter(timeline)
        steering = _SteeringState(input_adapter)

        first = steering.nudge_axis(
            axis="yaw",
            output=0.45,
            positive_key="d",
            negative_key="a",
            engage_threshold=0.22,
            release_threshold=0.08,
            hold_seconds=0.30,
            release_seconds=0.10,
            now=1.0,
        )
        second = steering.nudge_axis(
            axis="yaw",
            output=-0.45,
            positive_key="d",
            negative_key="a",
            engage_threshold=0.22,
            release_threshold=0.08,
            hold_seconds=0.30,
            release_seconds=0.10,
            now=1.1,
        )
        third = steering.nudge_axis(
            axis="yaw",
            output=-0.45,
            positive_key="d",
            negative_key="a",
            engage_threshold=0.22,
            release_threshold=0.08,
            hold_seconds=0.30,
            release_seconds=0.10,
            now=1.31,
        )

        self.assertEqual(first["transition"], "press:d")
        self.assertEqual(second["transition"], "pulse_hold")
        self.assertEqual(third["transition"], "release:d")
        self.assertEqual(timeline, [("key_down", "d"), ("key_up", "d")])


if __name__ == "__main__":
    unittest.main()
