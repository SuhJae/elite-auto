from __future__ import annotations

import logging
import unittest
from pathlib import Path

from app.actions.unload_construction import UnloadConstruction, UnloadConstructionTimings
from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import ShipState


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
        self.calls: list[tuple[str, object]] = []

    def set_throttle_percent(self, percent: int) -> None:
        self.calls.append(("throttle", percent))

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


def build_context(is_docked: bool, ship_control: FakeShipControl, input_adapter: FakeInputAdapter) -> Context:
    logger = logging.getLogger("elite_auto.test_unload_construction")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    state = ShipState(
        is_docked=is_docked,
        is_mass_locked=False,
        is_supercruise=False,
        cargo_count=0,
        gui_focus=None,
        status_flags=0,
    )
    return Context(
        config=AppConfig.default(),
        logger=logger,
        debug_snapshot_dir=Path("debug_snapshots"),
        state_reader=FakeStateReader(state),
        event_stream=FakeEventStream(),
        input_adapter=input_adapter,
        ship_control=ship_control,
    )


class TestUnloadConstruction(unittest.TestCase):
    def test_unload_construction_runs_requested_inputs(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(True, ship_control, input_adapter)
        action = UnloadConstruction(
            timings=UnloadConstructionTimings(
                menu_open_wait_seconds=0.0,
                scroll_down_hold_seconds=0.0,
                key_interval_seconds=0.0,
                post_confirm_wait_seconds=0.0,
            )
        )

        result = action.run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            ship_control.calls,
            [
                ("ui_select", "select"),
                ("ui_select", "up"),
                ("ui_select", "right"),
                ("ui_select", "right"),
                ("ui_select", "select"),
                ("ui_select", "left"),
                ("ui_select", "select"),
                ("ui_select", "down"),
            ],
        )
        self.assertEqual(
            input_adapter.calls,
            [
                ("hold", "down", 0.0),
                ("press", "backspace", 1.0),
            ],
        )

    def test_unload_construction_requires_docked_state(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(False, ship_control, input_adapter)
        action = UnloadConstruction(
            timings=UnloadConstructionTimings(
                menu_open_wait_seconds=0.0,
                scroll_down_hold_seconds=0.0,
                key_interval_seconds=0.0,
                post_confirm_wait_seconds=0.0,
            )
        )

        result = action.run(context)

        self.assertFalse(result.success)
        self.assertIn("undocked", result.reason.lower())


if __name__ == "__main__":
    unittest.main()
