from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.actions.navigation import LockNavDestination, NavigationTimings, OcrNavConfig, OcrNavTimings, OpenNavPanel
from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import ShipState


class SequenceStateReader:
    def __init__(self, states: list[ShipState]) -> None:
        self._states = states
        self._index = 0

    def snapshot(self) -> ShipState:
        state = self._states[min(self._index, len(self._states) - 1)]
        self._index += 1
        return state

    def is_docked(self) -> bool:
        return self.snapshot().is_docked

    def is_mass_locked(self) -> bool:
        return self.snapshot().is_mass_locked

    def is_supercruise(self) -> bool:
        return self.snapshot().is_supercruise

    def cargo_count(self) -> int:
        return self.snapshot().cargo_count

    def gui_focus(self) -> int | None:
        return self.snapshot().gui_focus


class FakeEventStream:
    def poll_events(self, limit: int | None = None) -> list[object]:
        return []


class FakeInputAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        self.calls.append(("press", key, float(presses)))

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", key, 0.0))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", key, 0.0))

    def hold(self, key: str, seconds: float) -> None:
        self.calls.append(("hold", key, seconds))


class FakeShipControl:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def set_throttle_percent(self, percent: int) -> None:
        self.calls.append(("set_throttle_percent", str(percent)))

    def boost(self) -> None:
        self.calls.append(("boost", None))

    def open_left_panel(self) -> None:
        self.calls.append(("open_left_panel", None))

    def cycle_previous_panel(self) -> None:
        self.calls.append(("cycle_previous_panel", None))

    def cycle_next_panel(self) -> None:
        self.calls.append(("cycle_next_panel", None))

    def ui_select(self, direction: str = "select") -> None:
        self.calls.append(("ui_select", direction))

    def charge_fsd(self, target: str = "supercruise") -> None:
        self.calls.append(("charge_fsd", target))


def build_state(*, gui_focus: int | None) -> ShipState:
    return ShipState(
        is_docked=False,
        is_mass_locked=False,
        is_supercruise=False,
        cargo_count=0,
        gui_focus=gui_focus,
        status_flags=0,
        raw_status={"GuiFocus": gui_focus} if gui_focus is not None else {},
    )


def build_context(states: list[ShipState], ship_control: FakeShipControl, input_adapter: FakeInputAdapter) -> Context:
    logger = logging.getLogger("elite_auto.test_navigation")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return Context(
        config=AppConfig.default(),
        logger=logger,
        debug_snapshot_dir=Path("debug_snapshots"),
        state_reader=SequenceStateReader(states),
        event_stream=FakeEventStream(),
        input_adapter=input_adapter,
        ship_control=ship_control,
    )


class TestNavigation(unittest.TestCase):
    def test_open_nav_panel_confirms_focus_cycles_previous_and_anchors_top(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(
            [build_state(gui_focus=None), build_state(gui_focus=2), build_state(gui_focus=2)],
            ship_control,
            input_adapter,
        )

        result = OpenNavPanel(
            timings=NavigationTimings(
                panel_open_confirm_timeout_seconds=0.01,
                panel_open_retry_count=1,
                panel_open_poll_interval_seconds=0.0,
                tab_previous_interval_seconds=0.0,
                anchor_hold_seconds=0.0,
                anchor_settle_seconds=0.0,
            )
        ).run(context)

        self.assertTrue(result.success)
        self.assertEqual(input_adapter.calls, [("hold", "up", 0.0)])
        self.assertEqual(
            ship_control.calls,
            [
                ("open_left_panel", None),
                ("cycle_previous_panel", None),
                ("cycle_previous_panel", None),
            ],
        )

    def test_lock_nav_destination_runs_open_ocr_select_select_back(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(
            [build_state(gui_focus=None), build_state(gui_focus=2), build_state(gui_focus=2)],
            ship_control,
            input_adapter,
        )

        with patch("app.actions.navigation.MoveCursorToNavTarget.run", return_value=type("R", (), {"success": True, "debug": {"target_name": "HIP 17189"}})()):
            result = LockNavDestination(
                target_name="HIP 17189",
                timings=NavigationTimings(
                    panel_open_confirm_timeout_seconds=0.01,
                    panel_open_retry_count=1,
                    panel_open_poll_interval_seconds=0.0,
                    tab_previous_interval_seconds=0.0,
                    anchor_hold_seconds=0.0,
                    anchor_settle_seconds=0.0,
                    lock_select_interval_seconds=0.0,
                    back_to_cockpit_wait_seconds=0.0,
                ),
            ).run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            ship_control.calls,
            [
                ("open_left_panel", None),
                ("cycle_previous_panel", None),
                ("cycle_previous_panel", None),
                ("ui_select", "select"),
                ("ui_select", "select"),
            ],
        )
        self.assertEqual(input_adapter.calls, [("hold", "up", 0.0), ("press", "backspace", 1.0)])

    def test_lock_nav_destination_runs_ocr_without_debug_artifacts(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(
            [build_state(gui_focus=None), build_state(gui_focus=2), build_state(gui_focus=2)],
            ship_control,
            input_adapter,
        )

        ocr_action = MagicMock()
        ocr_action.run.return_value = type("R", (), {"success": True, "debug": {"target_name": "HIP 17189"}})()

        with patch("app.actions.navigation.MoveCursorToNavTarget", return_value=ocr_action) as action_cls:
            result = LockNavDestination(
                target_name="HIP 17189",
                timings=NavigationTimings(
                    panel_open_confirm_timeout_seconds=0.01,
                    panel_open_retry_count=1,
                    panel_open_poll_interval_seconds=0.0,
                    tab_previous_interval_seconds=0.0,
                    anchor_hold_seconds=0.0,
                    anchor_settle_seconds=0.0,
                    lock_select_interval_seconds=0.0,
                    back_to_cockpit_wait_seconds=0.0,
                ),
            ).run(context)

        self.assertTrue(result.success)
        _, kwargs = action_cls.call_args
        self.assertEqual(kwargs["target_name"], "HIP 17189")
        self.assertEqual(kwargs["timings"], OcrNavTimings(move_interval_seconds=0.2))
        self.assertEqual(kwargs["config"], OcrNavConfig(save_debug_artifacts=False))


if __name__ == "__main__":
    unittest.main()
