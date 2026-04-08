from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import ShipState
from app.domain.result import Result
from colonize import ColonizeFsm, ColonizeState, parse_start_stage


class FakeStateReader:
    def snapshot(self) -> ShipState:
        return ShipState(
            is_docked=False,
            is_mass_locked=False,
            is_supercruise=False,
            cargo_count=0,
            gui_focus=None,
            status_flags=0,
        )


class FakeEventStream:
    def poll_events(self, limit: int | None = None) -> list[object]:
        return []


class FakeMarketDataSource:
    def snapshot(self, required: bool = False) -> object:
        raise AssertionError("The market data source should be handled by the patched buy action in this test.")


class RecordingAction:
    def __init__(self, name: str, result: Result, call_log: list[str]) -> None:
        self.name = name
        self._result = result
        self._call_log = call_log

    def run(self, context: Context) -> Result:
        self._call_log.append(self.name)
        return self._result


class FlakyRecordingAction:
    def __init__(self, name: str, results: list[Result], call_log: list[str]) -> None:
        self.name = name
        self._results = results
        self._call_log = call_log
        self._index = 0

    def run(self, context: Context) -> Result:
        self._call_log.append(self.name)
        result = self._results[min(self._index, len(self._results) - 1)]
        self._index += 1
        return result


def build_context() -> Context:
    logger = logging.getLogger("elite_auto.test_colonize")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return Context(
        config=AppConfig.default(),
        logger=logger,
        debug_snapshot_dir=Path("debug_snapshots"),
        state_reader=FakeStateReader(),
        event_stream=FakeEventStream(),
    )


class TestColonizeFsm(unittest.TestCase):
    def test_parse_start_stage_accepts_station_docking_alias(self) -> None:
        self.assertEqual(parse_start_stage("dock-station"), ColonizeState.DOCK_AT_REFINERY)

    def test_colonize_fsm_runs_expected_action_order(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = ColonizeFsm(
            station_name="HILDEBRANDT REFINERY",
            commodity_name="LIQUID OXYGEN",
            target_name="ORBITAL CONSTRUCTION SITE: PARISE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            cycle_limit=1,
            retry_attempts_per_state=1,
        )

        with (
            patch(
                "colonize.BuyFromStarport",
                return_value=RecordingAction("buy_from_starport", Result.ok("Bought cargo."), call_log),
            ),
            patch(
                "colonize.LeaveStation",
                return_value=RecordingAction("leave_station", Result.ok("Left station."), call_log),
            ),
            patch(
                "colonize.LockNavDestination",
                return_value=RecordingAction("lock_nav_destination", Result.ok("Locked target."), call_log),
            ),
            patch(
                "colonize.AlignToTargetCompass",
                return_value=RecordingAction("align_to_target_compass", Result.ok("Aligned ship."), call_log),
            ),
            patch(
                "colonize.EngageFsdSequence",
                return_value=RecordingAction("engage_fsd_sequence", Result.ok("Engaged FSD."), call_log),
            ),
            patch(
                "colonize.RequestDockingSequence",
                return_value=RecordingAction("request_docking_sequence", Result.ok("Requested docking."), call_log),
            ),
            patch(
                "colonize.UnloadConstruction",
                return_value=RecordingAction("unload_construction", Result.ok("Unloaded construction."), call_log),
            ),
            patch(
                "colonize.LeaveConstruction",
                return_value=RecordingAction("leave_construction", Result.ok("Left construction."), call_log),
            ),
        ):
            result = fsm.run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            call_log,
            [
                "buy_from_starport",
                "leave_station",
                "lock_nav_destination",
                "align_to_target_compass",
                "engage_fsd_sequence",
                "request_docking_sequence",
                "unload_construction",
                "leave_construction",
                "lock_nav_destination",
                "align_to_target_compass",
                "engage_fsd_sequence",
                "align_to_target_compass",
                "request_docking_sequence",
            ],
        )

    def test_colonize_fsm_stops_after_first_failed_state(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = ColonizeFsm(
            station_name="HILDEBRANDT REFINERY",
            commodity_name="LIQUID OXYGEN",
            target_name="ORBITAL CONSTRUCTION SITE: PARISE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            cycle_limit=1,
            retry_attempts_per_state=1,
        )

        with (
            patch(
                "colonize.BuyFromStarport",
                return_value=RecordingAction("buy_from_starport", Result.ok("Bought cargo."), call_log),
            ),
            patch(
                "colonize.LeaveStation",
                return_value=RecordingAction("leave_station", Result.fail("Auto-launch failed."), call_log),
            ),
            patch("colonize.LockNavDestination"),
            patch("colonize.AlignToTargetCompass"),
            patch("colonize.EngageFsdSequence"),
            patch("colonize.RequestDockingSequence"),
            patch("colonize.UnloadConstruction"),
            patch("colonize.LeaveConstruction"),
        ):
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertEqual(call_log, ["buy_from_starport", "leave_station"])
        self.assertIn("leave_station", result.reason)

    def test_colonize_fsm_can_start_from_requested_stage(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = ColonizeFsm(
            station_name="HILDEBRANDT REFINERY",
            commodity_name="LIQUID OXYGEN",
            target_name="ORBITAL CONSTRUCTION SITE: PARISE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            cycle_limit=1,
            initial_state=ColonizeState.DOCK_AT_REFINERY,
            retry_attempts_per_state=1,
        )

        with patch(
            "colonize.RequestDockingSequence",
            return_value=RecordingAction("request_docking_sequence", Result.ok("Requested docking."), call_log),
        ):
            result = fsm.run(context)

        self.assertTrue(result.success)
        self.assertEqual(call_log, ["request_docking_sequence"])

    def test_colonize_fsm_retries_failed_state_until_success(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = ColonizeFsm(
            station_name="HILDEBRANDT REFINERY",
            commodity_name="LIQUID OXYGEN",
            target_name="ORBITAL CONSTRUCTION SITE: PARISE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            cycle_limit=1,
            initial_state=ColonizeState.DOCK_AT_REFINERY,
            retry_attempts_per_state=5,
        )

        flaky_action = FlakyRecordingAction(
            "request_docking_sequence",
            [Result.fail("first"), Result.fail("second"), Result.ok("done")],
            call_log,
        )

        with patch("colonize.RequestDockingSequence", return_value=flaky_action), patch(
            "colonize.time.sleep", return_value=None
        ):
            result = fsm.run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            call_log,
            ["request_docking_sequence", "request_docking_sequence", "request_docking_sequence"],
        )


if __name__ == "__main__":
    unittest.main()
