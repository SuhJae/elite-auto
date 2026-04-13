from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import ShipState
from app.domain.result import Result
from trade import TradeFsm, TradeState, parse_start_stage


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
        raise AssertionError("The market data source should be handled by patched actions in this test.")


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
    logger = logging.getLogger("elite_auto.test_trade")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return Context(
        config=AppConfig.default(),
        logger=logger,
        debug_snapshot_dir=Path("debug_snapshots"),
        state_reader=FakeStateReader(),
        event_stream=FakeEventStream(),
    )


class TestTradeFsm(unittest.TestCase):
    def test_parse_start_stage_accepts_source_docking_alias(self) -> None:
        self.assertEqual(parse_start_stage("dock-source"), TradeState.DOCK_AT_SOURCE)

    def test_trade_fsm_defaults_carrier_market_name_to_callsign(self) -> None:
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER BZL-59X",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            source_is_carrier=True,
        )

        self.assertEqual(fsm.source_market_name, "BZL-59X")

    def test_trade_fsm_passes_source_carrier_flag_to_buy_action_when_enabled(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            source_is_carrier=True,
            cycle_limit=1,
            initial_state=TradeState.BUY_FROM_SOURCE,
            retry_attempts_per_state=1,
        )

        with (
            patch(
                "trade.BuyFromStarport",
                return_value=RecordingAction("buy_from_starport", Result.ok("Bought cargo."), call_log),
            ),
            patch(
                "trade.LeaveStation",
                return_value=RecordingAction("leave_station", Result.fail("stop"), call_log),
            ),
        ):
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertEqual(call_log, ["buy_from_starport", "leave_station"])

    def test_trade_fsm_uses_source_market_name_for_buy_station_check(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER BZL-59X",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            source_is_carrier=True,
            source_market_name="BZL-59X",
            cycle_limit=1,
            initial_state=TradeState.BUY_FROM_SOURCE,
            retry_attempts_per_state=1,
        )

        captured_kwargs: dict[str, object] = {}

        def build_buy_action(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return RecordingAction("buy_from_starport", Result.fail("stop"), [])

        with patch("trade.BuyFromStarport", side_effect=build_buy_action):
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertEqual(captured_kwargs["station_name"], "BZL-59X")
        self.assertEqual(captured_kwargs["max_buy_price"], 15500)

    def test_trade_fsm_passes_top_item_flag_to_destination_sell(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER BZL-59X",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            destination_item_is_top=True,
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=1,
        )

        captured_kwargs: dict[str, object] = {}

        def build_sell_action(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return RecordingAction("sell_from_starport", Result.fail("stop"), [])

        with patch("trade.SellFromStarport", side_effect=build_sell_action):
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertEqual(captured_kwargs["is_top"], True)

    def test_trade_fsm_runs_expected_action_order(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            cycle_limit=1,
            retry_attempts_per_state=1,
        )

        with (
            patch(
                "trade.BuyFromStarport",
                return_value=RecordingAction("buy_from_starport", Result.ok("Bought cargo."), call_log),
            ),
            patch(
                "trade.LeaveStation",
                return_value=RecordingAction("leave_station", Result.ok("Left station."), call_log),
            ),
            patch(
                "trade.LockNavDestination",
                return_value=RecordingAction("lock_nav_destination", Result.ok("Locked target."), call_log),
            ),
            patch(
                "trade.AlignToTargetCompass",
                return_value=RecordingAction("align_to_target_compass", Result.ok("Aligned ship."), call_log),
            ),
            patch(
                "trade.EngageFsdSequence",
                return_value=RecordingAction("engage_fsd_sequence", Result.ok("Engaged FSD."), call_log),
            ),
            patch(
                "trade.RequestDockingSequence",
                return_value=RecordingAction("request_docking_sequence", Result.ok("Requested docking."), call_log),
            ),
            patch(
                "trade.SellFromStarport",
                return_value=RecordingAction("sell_from_starport", Result.ok("Sold cargo."), call_log),
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
                "sell_from_starport",
                "leave_station",
                "lock_nav_destination",
                "align_to_target_compass",
                "engage_fsd_sequence",
                "request_docking_sequence",
            ],
        )

    def test_trade_fsm_stops_after_first_failed_state(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            cycle_limit=1,
            retry_attempts_per_state=1,
        )

        with (
            patch(
                "trade.BuyFromStarport",
                return_value=RecordingAction("buy_from_starport", Result.ok("Bought cargo."), call_log),
            ),
            patch(
                "trade.LeaveStation",
                return_value=RecordingAction("leave_station", Result.fail("Auto-launch failed."), call_log),
            ),
            patch("trade.LockNavDestination"),
            patch("trade.AlignToTargetCompass"),
            patch("trade.EngageFsdSequence"),
            patch("trade.RequestDockingSequence"),
            patch("trade.SellFromStarport"),
        ):
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertEqual(call_log, ["buy_from_starport", "leave_station"])
        self.assertIn("leave_source", result.reason)

    def test_trade_fsm_can_start_from_requested_stage(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=1,
        )

        with (
            patch(
                "trade.SellFromStarport",
                return_value=RecordingAction("sell_from_starport", Result.ok("Sold cargo."), call_log),
            ),
            patch(
                "trade.LeaveStation",
                return_value=RecordingAction("leave_station", Result.ok("Left station."), call_log),
            ),
            patch(
                "trade.LockNavDestination",
                return_value=RecordingAction("lock_nav_destination", Result.ok("Locked target."), call_log),
            ),
            patch(
                "trade.AlignToTargetCompass",
                return_value=RecordingAction("align_to_target_compass", Result.ok("Aligned ship."), call_log),
            ),
            patch(
                "trade.EngageFsdSequence",
                return_value=RecordingAction("engage_fsd_sequence", Result.ok("Engaged FSD."), call_log),
            ),
            patch(
                "trade.RequestDockingSequence",
                return_value=RecordingAction("request_docking_sequence", Result.ok("Requested docking."), call_log),
            ),
        ):
            result = fsm.run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            call_log,
            [
                "sell_from_starport",
                "leave_station",
                "lock_nav_destination",
                "align_to_target_compass",
                "engage_fsd_sequence",
                "request_docking_sequence",
            ],
        )

    def test_trade_fsm_retries_failed_state_until_success(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            cycle_limit=1,
            initial_state=TradeState.DOCK_AT_SOURCE,
            retry_attempts_per_state=5,
        )

        flaky_action = FlakyRecordingAction(
            "request_docking_sequence",
            [Result.fail("first"), Result.fail("second"), Result.ok("done")],
            call_log,
        )

        with patch("trade.RequestDockingSequence", return_value=flaky_action), patch(
            "trade.time.sleep", return_value=None
        ):
            result = fsm.run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            call_log,
            ["request_docking_sequence", "request_docking_sequence", "request_docking_sequence"],
        )


if __name__ == "__main__":
    unittest.main()
