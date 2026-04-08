from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.actions.align_nav_beacon import (
    AlignToNavBeacon,
    NavBeaconAlignConfig,
    NavBeaconDetection,
    _select_alignment_command,
)
from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import ShipState


class FakeStateReader:
    def __init__(self, state: ShipState) -> None:
        self._state = state

    def snapshot(self) -> ShipState:
        return self._state


class FakeEventStream:
    def poll_events(self, limit: int | None = None) -> list[object]:
        return []


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], timeline: list[tuple[str, object]]) -> None:
        self._frames = frames
        self._timeline = timeline
        self._index = 0

    def grab(self, region=None):  # noqa: ANN001 - test double
        self._timeline.append(("grab", region))
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        return frame.copy()


class FakeInputAdapter:
    def __init__(self, timeline: list[tuple[str, object]]) -> None:
        self._timeline = timeline
        self.holds: list[tuple[str, float]] = []

    def hold(self, key: str, seconds: float) -> None:
        self.holds.append((key, seconds))
        self._timeline.append(("hold", (key, seconds)))


class FakeVision:
    def __init__(self, root: Path) -> None:
        self._root = root
        self.saved_paths: list[Path] = []

    def save_debug_snapshot(self, name: str, image) -> Path:  # noqa: ANN001
        path = self._root / f"{name}.png"
        path.write_bytes(b"fake")
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
) -> tuple[Context, FakeInputAdapter]:
    logger = logging.getLogger("elite_auto.test_align_nav_beacon")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    config = AppConfig.default()
    input_adapter = FakeInputAdapter(timeline)
    context = Context(
        config=config,
        logger=logger,
        debug_snapshot_dir=debug_snapshot_dir,
        state_reader=FakeStateReader(build_state(is_docked=False)),
        event_stream=FakeEventStream(),
        input_adapter=input_adapter,
        ship_control=None,
        capture=FakeCapture(frames, timeline),
        vision=FakeVision(debug_snapshot_dir),
    )
    return context, input_adapter


class TestNavBeaconAlign(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NavBeaconAlignConfig(template_path="fake-template.png")
        self.window_patch = patch("app.actions.align_nav_beacon._find_window_client_region", return_value=(0, 0, 1920, 1080))
        self.output_patch = patch("app.actions.align_nav_beacon._resolve_output_region", return_value=(0, (0, 0, 1920, 1080)))
        self.template_patch = patch(
            "app.actions.align_nav_beacon._load_template_image",
            return_value=np.zeros((40, 40, 3), dtype=np.uint8),
        )
        self.window_patch.start()
        self.output_patch.start()
        self.template_patch.start()
        self.addCleanup(self.window_patch.stop)
        self.addCleanup(self.output_patch.stop)
        self.addCleanup(self.template_patch.stop)

    def test_select_alignment_command_prefers_yaw_for_horizontal_error(self) -> None:
        detection = NavBeaconDetection(
            detected=True,
            center_x=920.0,
            center_y=540.0,
            dx=-40.0,
            dy=5.0,
            status="detected",
        )

        command = _select_alignment_command(detection, self.config)

        self.assertEqual(command.axis, "yaw")
        self.assertEqual(command.key_name, "yaw_left")
        self.assertGreater(command.pulse_seconds, 0.0)

    def test_select_alignment_command_prefers_pitch_for_vertical_error(self) -> None:
        detection = NavBeaconDetection(
            detected=True,
            center_x=960.0,
            center_y=480.0,
            dx=3.0,
            dy=-60.0,
            status="detected",
        )

        command = _select_alignment_command(detection, self.config)

        self.assertEqual(command.axis, "pitch")
        self.assertEqual(command.key_name, "pitch_up")
        self.assertGreater(command.pulse_seconds, 0.0)

    def test_align_succeeds_after_three_centered_reads(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)] * 3

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter = build_context(frames, timeline, Path(temp_dir))
            reads = [
                NavBeaconDetection(detected=True, center_x=960.0, center_y=540.0, dx=0.0, dy=0.0, status="detected"),
                NavBeaconDetection(detected=True, center_x=961.0, center_y=539.0, dx=1.0, dy=-1.0, status="detected"),
                NavBeaconDetection(detected=True, center_x=959.0, center_y=541.0, dx=-1.0, dy=1.0, status="detected"),
            ]

            with patch("app.actions.align_nav_beacon.detect_nav_beacon", side_effect=reads), patch(
                "app.actions.align_nav_beacon.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToNavBeacon(config=self.config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(input_adapter.holds, [])

    def test_align_emits_hold_when_beacon_is_off_center(self) -> None:
        timeline: list[tuple[str, object]] = []
        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)] * 4

        with tempfile.TemporaryDirectory() as temp_dir:
            context, input_adapter = build_context(frames, timeline, Path(temp_dir))
            reads = [
                NavBeaconDetection(detected=True, center_x=900.0, center_y=540.0, dx=-60.0, dy=0.0, status="detected"),
                NavBeaconDetection(detected=True, center_x=960.0, center_y=540.0, dx=0.0, dy=0.0, status="detected"),
                NavBeaconDetection(detected=True, center_x=960.0, center_y=540.0, dx=0.0, dy=0.0, status="detected"),
                NavBeaconDetection(detected=True, center_x=960.0, center_y=540.0, dx=0.0, dy=0.0, status="detected"),
            ]

            with patch("app.actions.align_nav_beacon.detect_nav_beacon", side_effect=reads), patch(
                "app.actions.align_nav_beacon.time.sleep", side_effect=lambda seconds: timeline.append(("sleep", seconds))
            ):
                result = AlignToNavBeacon(config=self.config).run(context)

        self.assertTrue(result.success)
        self.assertEqual(len(input_adapter.holds), 1)
        self.assertEqual(input_adapter.holds[0][0], context.config.controls.yaw_left)


if __name__ == "__main__":
    unittest.main()
