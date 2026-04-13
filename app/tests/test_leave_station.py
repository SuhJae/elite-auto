from __future__ import annotations

import logging
import unittest
from pathlib import Path

from app.actions.leave_station import LeaveStation, LeaveStationTimings
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


class FakeShipControl:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def set_throttle_percent(self, percent: int) -> None:
        self.calls.append(("throttle", percent))

    def boost(self) -> None:
        self.calls.append(("boost", None))

    def open_left_panel(self) -> None:
        self.calls.append(("open_left_panel", None))

    def cycle_next_panel(self) -> None:
        self.calls.append(("cycle_next_panel", None))

    def ui_select(self, direction: str = "select") -> None:
        self.calls.append(("ui_select", direction))

    def charge_fsd(self, target: str = "supercruise") -> None:
        self.calls.append(("charge_fsd", target))


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


def build_state(*, is_docked: bool, is_mass_locked: bool) -> ShipState:
    return ShipState(
        is_docked=is_docked,
        is_mass_locked=is_mass_locked,
        is_supercruise=False,
        cargo_count=0,
        gui_focus=None,
        status_flags=0,
    )


def build_context(states: list[ShipState], ship_control: FakeShipControl, input_adapter: FakeInputAdapter) -> Context:
    logger = logging.getLogger("elite_auto.test_leave_station")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    config = AppConfig.default()
    config.controls.thrust_up = "r"
    return Context(
        config=config,
        logger=logger,
        debug_snapshot_dir=Path("debug_snapshots"),
        state_reader=SequenceStateReader(states),
        event_stream=FakeEventStream(),
        input_adapter=input_adapter,
        ship_control=ship_control,
    )


class TestLeaveStation(unittest.TestCase):
    def test_leave_station_runs_launch_and_thrust_up_sequence(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(
            [
                build_state(is_docked=True, is_mass_locked=True),
                build_state(is_docked=False, is_mass_locked=True),
                build_state(is_docked=False, is_mass_locked=True),
                build_state(is_docked=False, is_mass_locked=False),
                build_state(is_docked=False, is_mass_locked=False),
            ],
            ship_control,
            input_adapter,
        )
        action = LeaveStation(
            timings=LeaveStationTimings(
                auto_launch_wait_seconds=0.0,
                mass_lock_poll_interval_seconds=0.0,
                post_mass_lock_clear_wait_seconds=0.0,
                mass_lock_timeout_seconds=1.0,
            )
        )

        result = action.run(context)

        self.assertTrue(result.success)
        self.assertEqual(ship_control.calls, [("ui_select", "select"), ("throttle", 100), ("throttle", 0)])
        self.assertEqual(input_adapter.calls, [("key_down", "r", 0.0), ("key_up", "r", 0.0)])

    def test_leave_station_rejects_already_undocked(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context([build_state(is_docked=False, is_mass_locked=False)], ship_control, input_adapter)
        action = LeaveStation(
            timings=LeaveStationTimings(
                auto_launch_wait_seconds=0.0,
                mass_lock_poll_interval_seconds=0.0,
                post_mass_lock_clear_wait_seconds=0.0,
                mass_lock_timeout_seconds=1.0,
            )
        )

        result = action.run(context)

        self.assertFalse(result.success)
        self.assertIn("already undocked", result.reason.lower())

    def test_leave_station_releases_thrust_when_mass_lock_wait_fails(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(
            [
                build_state(is_docked=True, is_mass_locked=True),
                build_state(is_docked=False, is_mass_locked=True),
                build_state(is_docked=False, is_mass_locked=True),
            ],
            ship_control,
            input_adapter,
        )
        action = LeaveStation(
            timings=LeaveStationTimings(
                auto_launch_wait_seconds=0.0,
                mass_lock_poll_interval_seconds=0.0,
                post_mass_lock_clear_wait_seconds=0.0,
                mass_lock_timeout_seconds=0.0,
            )
        )

        result = action.run(context)

        self.assertFalse(result.success)
        self.assertEqual(ship_control.calls, [("ui_select", "select"), ("throttle", 100)])
        self.assertEqual(input_adapter.calls, [("key_down", "r", 0.0), ("key_up", "r", 0.0)])


if __name__ == "__main__":
    unittest.main()
