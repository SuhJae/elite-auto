from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from app.actions.lock_galaxy import GalaxyLockConfig, GalaxyLockTimings, LockGalaxy
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
        return None

    def boost(self) -> None:
        return None

    def open_left_panel(self) -> None:
        return None

    def cycle_previous_panel(self) -> None:
        return None

    def cycle_next_panel(self) -> None:
        return None

    def ui_select(self, direction: str = "select") -> None:
        self.calls.append(("ui_select", direction))

    def charge_fsd(self, target: str = "supercruise") -> None:
        return None


def build_state(*, gui_focus: int | None, destination_name: str | None = None) -> ShipState:
    raw_status = {"GuiFocus": gui_focus} if gui_focus is not None else {}
    if destination_name is not None:
        raw_status["Destination"] = {"Name": destination_name}
    return ShipState(
        is_docked=False,
        is_mass_locked=False,
        is_supercruise=False,
        cargo_count=0,
        gui_focus=gui_focus,
        status_flags=0,
        raw_status=raw_status,
    )


def build_context(states: list[ShipState], ship_control: FakeShipControl, input_adapter: FakeInputAdapter) -> Context:
    logger = logging.getLogger("elite_auto.test_lock_galaxy")
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


class TestLockGalaxy(unittest.TestCase):
    def test_lock_galaxy_opens_map_types_target_confirms_destination_and_closes(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(
            [
                build_state(gui_focus=0),
                build_state(gui_focus=6),
                build_state(gui_focus=6, destination_name="BODEDI"),
                build_state(gui_focus=0, destination_name="BODEDI"),
                build_state(gui_focus=0, destination_name="BODEDI"),
            ],
            ship_control,
            input_adapter,
        )

        with patch("app.actions.lock_galaxy._move_mouse_to_window_center_and_hold", return_value=True):
            result = LockGalaxy(
                target_name="BODEDI",
                timings=GalaxyLockTimings(
                    map_open_timeout_seconds=0.01,
                    map_poll_interval_seconds=0.0,
                    map_retry_count=1,
                    post_map_open_delay_seconds=0.0,
                    key_interval_seconds=0.0,
                    search_settle_seconds=0.0,
                    post_result_select_delay_seconds=0.0,
                    click_hold_seconds=0.0,
                    destination_confirm_timeout_seconds=0.01,
                    close_timeout_seconds=0.01,
                    close_delete_presses=2,
                    close_extra_delete_presses=1,
                ),
                config=GalaxyLockConfig(window_title_substring="Elite Dangerous", open_key="f4"),
            ).run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            ship_control.calls,
            [
                ("ui_select", "select"),
                ("ui_select", "down"),
                ("ui_select", "select"),
            ],
        )
        self.assertEqual(
            input_adapter.calls,
            [
                ("press", "f4", 1.0),
                ("press", "up", 1.0),
                ("press", "b", 1.0),
                ("press", "o", 1.0),
                ("press", "d", 1.0),
                ("press", "e", 1.0),
                ("press", "d", 1.0),
                ("press", "i", 1.0),
                ("press", "backspace", 2.0),
            ],
        )

    def test_lock_galaxy_fails_safe_when_destination_does_not_update(self) -> None:
        ship_control = FakeShipControl()
        input_adapter = FakeInputAdapter()
        context = build_context(
            [
                build_state(gui_focus=0),
                build_state(gui_focus=6),
                build_state(gui_focus=6),
                build_state(gui_focus=0),
                build_state(gui_focus=0),
            ],
            ship_control,
            input_adapter,
        )

        with patch("app.actions.lock_galaxy._move_mouse_to_window_center_and_hold", return_value=True):
            result = LockGalaxy(
                target_name="BODEDI",
                timings=GalaxyLockTimings(
                    map_open_timeout_seconds=0.01,
                    map_poll_interval_seconds=0.0,
                    map_retry_count=1,
                    post_map_open_delay_seconds=0.0,
                    key_interval_seconds=0.0,
                    search_settle_seconds=0.0,
                    post_result_select_delay_seconds=0.0,
                    click_hold_seconds=0.0,
                    destination_confirm_timeout_seconds=0.01,
                    close_timeout_seconds=0.01,
                    close_delete_presses=2,
                    close_extra_delete_presses=1,
                ),
            ).run(context)

        self.assertFalse(result.success)
        self.assertIn("failed to lock galaxy destination", result.reason.lower())
        self.assertIn(("press", "backspace", 2.0), input_adapter.calls)


if __name__ == "__main__":
    unittest.main()
