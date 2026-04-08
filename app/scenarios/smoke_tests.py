from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.actions.fsd import WaitForMassLockClear
from app.actions.launch import WaitUntilUndocked
from app.actions.navigation import WaitForSupercruiseEntry
from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import JournalEvent, ShipState
from app.domain.result import Result
from app.routines.source_cycle import SampleDepartureRoutine


@dataclass
class FakeStateReader:
    """Test double that returns a predetermined sequence of ship states."""

    states: list[ShipState]

    def __post_init__(self) -> None:
        self._index = 0

    def snapshot(self) -> ShipState:
        if self._index < len(self.states):
            state = self.states[self._index]
            self._index += 1
            return state
        return self.states[-1]

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
    def poll_events(self, limit: int | None = None) -> list[JournalEvent]:
        return []


def build_fake_context(states: list[ShipState]) -> Context:
    logger = logging.getLogger("elite_auto.smoke")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    config = AppConfig.default()
    return Context(
        config=config,
        logger=logger,
        debug_snapshot_dir=Path("debug_snapshots"),
        state_reader=FakeStateReader(states=states),
        event_stream=FakeEventStream(),
    )


def run_action_smoke_tests() -> dict[str, Result]:
    """Run safe polling actions against fake state instead of the live game."""

    undocked_states = [
        ShipState(True, False, False, 0, None, 1),
        ShipState(False, False, False, 0, None, 0),
    ]
    mass_lock_states = [
        ShipState(False, True, False, 0, None, 0),
        ShipState(False, False, False, 0, None, 0),
    ]
    supercruise_states = [
        ShipState(False, False, False, 0, None, 0),
        ShipState(False, False, True, 0, None, 0),
    ]

    return {
        WaitUntilUndocked.name: WaitUntilUndocked(0.1, 0.0).run(build_fake_context(undocked_states)),
        WaitForMassLockClear.name: WaitForMassLockClear(0.1, 0.0).run(build_fake_context(mass_lock_states)),
        WaitForSupercruiseEntry.name: WaitForSupercruiseEntry(0.1, 0.0).run(
            build_fake_context(supercruise_states)
        ),
    }


def run_routine_smoke_test() -> Result:
    """Run the sample routine against a predictable fake state sequence."""

    states = [
        ShipState(True, True, False, 0, None, 0),
        ShipState(False, True, False, 0, None, 0),
        ShipState(False, False, False, 0, None, 0),
        ShipState(False, False, True, 0, None, 0),
    ]
    routine = SampleDepartureRoutine(timeout_seconds=0.1, poll_interval_seconds=0.0)
    return routine.run(build_fake_context(states))


def adapter_construction_smoke() -> dict[str, str]:
    """Smoke-check importable adapter classes without starting live integrations."""

    from app.adapters.capture_dxcam import DxcamCapture
    from app.adapters.files_watchdog import WatchdogFileWatcher
    from app.adapters.input_pydirect import PydirectInputAdapter
    from app.adapters.vision_cv import OpenCVVisionSystem

    return {
        "capture": DxcamCapture.__name__,
        "watcher": WatchdogFileWatcher.__name__,
        "input": PydirectInputAdapter.__name__,
        "vision": OpenCVVisionSystem.__name__,
    }
