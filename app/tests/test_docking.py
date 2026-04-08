from __future__ import annotations

import logging
import unittest
from pathlib import Path

from app.actions.docking import DockingTimings, RequestDockingSequence
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


def build_state(*, is_docked: bool) -> ShipState:
    return ShipState(
        is_docked=is_docked,
        is_mass_locked=False,
        is_supercruise=False,
        cargo_count=0,
        gui_focus=None,
        status_flags=0,
    )


def build_context(states: list[ShipState], ship_control: FakeShipControl, input_adapter: FakeInputAdapter) -> Context:
    logger = logging.getLogger("elite_auto.test_docking")
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


class TestDocking(unittest.TestCase):
    def test_request_docking_sequence_runs_boost_and_menu_steps(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(
            [
                build_state(is_docked=False),
                build_state(is_docked=False),
                build_state(is_docked=True),
                build_state(is_docked=False),
            ],
            ship_control,
            input_adapter,
        )
        action = RequestDockingSequence(
            timings=DockingTimings(
                pre_boost_wait_seconds=0.0,
                post_boost_wait_seconds=0.0,
                key_interval_seconds=0.0,
                post_select_back_wait_seconds=0.0,
                landing_confirm_timeout_seconds=1.0,
                post_docked_wait_seconds=0.0,
                poll_interval_seconds=0.0,
            )
        )

        result = action.run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            ship_control.calls,
            [
                ("boost", None),
                ("open_left_panel", None),
                ("cycle_next_panel", None),
                ("cycle_next_panel", None),
                ("ui_select", "right"),
                ("ui_select", "select"),
            ],
        )
        self.assertEqual(input_adapter.calls, [("press", "backspace", 1.0)])

    def test_request_docking_sequence_fails_when_landing_not_confirmed(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(
            [
                build_state(is_docked=False),
                build_state(is_docked=False),
                build_state(is_docked=False),
            ],
            ship_control,
            input_adapter,
        )
        action = RequestDockingSequence(
            timings=DockingTimings(
                pre_boost_wait_seconds=0.0,
                post_boost_wait_seconds=0.0,
                key_interval_seconds=0.0,
                post_select_back_wait_seconds=0.0,
                landing_confirm_timeout_seconds=0.0,
                post_docked_wait_seconds=0.0,
                poll_interval_seconds=0.0,
            )
        )

        result = action.run(context)

        self.assertFalse(result.success)
        self.assertIn("landing was not confirmed", result.reason.lower())

    def test_request_docking_sequence_rejects_already_docked(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context([build_state(is_docked=True)], ship_control, input_adapter)
        action = RequestDockingSequence(
            timings=DockingTimings(
                pre_boost_wait_seconds=0.0,
                post_boost_wait_seconds=0.0,
                key_interval_seconds=0.0,
                post_select_back_wait_seconds=0.0,
                landing_confirm_timeout_seconds=1.0,
                post_docked_wait_seconds=0.0,
                poll_interval_seconds=0.0,
            )
        )

        result = action.run(context)

        self.assertFalse(result.success)
        self.assertIn("already docked", result.reason.lower())


if __name__ == "__main__":
    unittest.main()
