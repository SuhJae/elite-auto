from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import CommodityListing, MarketSnapshot, ShipState
from app.domain.result import Result
from trade import DEFAULT_SAFETY_CONFIG, DEFAULT_SOURCE_CONFIG, TradeFsm, TradeState, parse_start_stage


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


class SnapshotMarketDataSource:
    def __init__(self, snapshot: MarketSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self, required: bool = False) -> MarketSnapshot:
        return self._snapshot


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


class ValidationAwareSellAction:
    def __init__(self, result: Result, market_data_source, market_validation) -> None:
        self.name = "sell_from_starport"
        self._result = result
        self._market_data_source = market_data_source
        self._market_validation = market_validation

    def run(self, context: Context) -> Result:
        if self._market_validation is not None:
            validation_result = self._market_validation(self._market_data_source.snapshot(required=True))
            if validation_result is not None and not validation_result.success:
                return validation_result
        return self._result


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


def build_market_snapshot(sell_price: int = 15000, demand: int = 1) -> MarketSnapshot:
    return MarketSnapshot(
        station_name="MILLE GATEWAY",
        station_type="Coriolis Starport",
        star_system="Fengiri",
        market_id=42,
        timestamp="2026-04-13T00:00:00Z",
        commodities=[
            CommodityListing(
                commodity_id=1,
                name="agronomictreatment",
                name_localised="Agronomic Treatment",
                category="chemicals",
                category_localised="Chemicals",
                buy_price=0,
                sell_price=sell_price,
                mean_price=10000,
                stock=0,
                demand=demand,
                stock_bracket=0,
                demand_bracket=3,
                consumer=True,
                producer=False,
                rare=False,
            )
        ],
        source_path=Path("Market.json"),
        source_timestamp=None,
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
        self.assertEqual(captured_kwargs["max_buy_price"], DEFAULT_SAFETY_CONFIG["source_max_buy_price"])

    def test_trade_fsm_passes_top_item_flag_to_destination_sell(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER BZL-59X",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=15000)),
            destination_item_is_top=True,
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=1,
        )
        fsm.last_source_buy_price = 10000

        captured_kwargs: dict[str, object] = {}

        def build_sell_action(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return RecordingAction("sell_from_starport", Result.fail("stop"), [])

        with patch("trade.SellFromStarport", side_effect=build_sell_action):
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertEqual(captured_kwargs["is_top"], True)

    def test_trade_fsm_passes_destination_carrier_flag_to_sell(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER BZL-59X",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=15000)),
            destination_is_carrier=True,
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=1,
        )
        fsm.last_source_buy_price = 10000

        captured_kwargs: dict[str, object] = {}

        def build_sell_action(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return RecordingAction("sell_from_starport", Result.fail("stop"), [])

        with patch("trade.SellFromStarport", side_effect=build_sell_action):
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertEqual(captured_kwargs["is_carrier"], True)

    def test_trade_fsm_defaults_destination_carrier_market_name_to_callsign(self) -> None:
        fsm = TradeFsm(
            source_station_name="Khun Port",
            commodity_name="Steel",
            destination_name="[BKRN] Event Horizon X5W-54J",
            market_data_source=FakeMarketDataSource(),
            destination_is_carrier=True,
        )

        self.assertEqual(fsm.destination_market_name, "X5W-54J")

    def test_trade_fsm_uses_destination_market_name_for_sell_station_check(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="Khun Port",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="[BKRN] Event Horizon X5W-54J",
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=15000)),
            destination_is_carrier=True,
            destination_market_name="X5W-54J",
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=1,
        )
        fsm.last_source_buy_price = 10000

        captured_kwargs: dict[str, object] = {}

        def build_sell_action(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return RecordingAction("sell_from_starport", Result.fail("stop"), [])

        with patch("trade.SellFromStarport", side_effect=build_sell_action):
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertEqual(captured_kwargs["station_name"], "X5W-54J")

    def test_trade_fsm_captures_unit_buy_price_after_buy(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            cycle_limit=1,
            initial_state=TradeState.BUY_FROM_SOURCE,
            retry_attempts_per_state=1,
        )

        with (
            patch(
                "trade.BuyFromStarport",
                return_value=RecordingAction(
                    "buy_from_starport",
                    Result.ok("Bought cargo.", debug={"unit_buy_price": 12000}),
                    [],
                ),
            ),
            patch(
                "trade.LeaveStation",
                return_value=RecordingAction("leave_station", Result.fail("stop"), []),
            ),
        ):
            fsm.run(context)

        self.assertEqual(fsm.last_source_buy_price, 12000)

    def test_trade_fsm_halts_sell_when_profit_below_minimum(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=14999)),
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=1,
            min_profit_per_unit=3000,
        )
        fsm.last_source_buy_price = 12000

        with patch(
            "trade.SellFromStarport",
            side_effect=lambda **kwargs: ValidationAwareSellAction(Result.ok("Sold cargo."), kwargs["market_data_source"], kwargs.get("market_validation")),
        ) as sell_action:
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertIn("below the configured minimum", result.reason)
        sell_action.assert_called_once()

    def test_trade_fsm_halts_sell_when_station_demand_below_minimum(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="Khun Port",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=20000, demand=999)),
            destination_is_carrier=False,
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=1,
            min_profit_per_unit=3000,
            destination_min_demand=1000,
        )
        fsm.last_source_buy_price = 12000

        with patch(
            "trade.SellFromStarport",
            side_effect=lambda **kwargs: ValidationAwareSellAction(Result.ok("Sold cargo."), kwargs["market_data_source"], kwargs.get("market_validation")),
        ) as sell_action:
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertIn("demand is below the configured minimum", result.reason)
        sell_action.assert_called_once()

    def test_trade_fsm_ignores_station_demand_floor_for_carrier_destination(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="Khun Port",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="[BKRN] Event Horizon X5W-54J",
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=20000, demand=0)),
            destination_is_carrier=True,
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=1,
            min_profit_per_unit=3000,
            destination_min_demand=1000,
        )
        fsm.last_source_buy_price = 12000

        with patch(
            "trade.SellFromStarport",
            return_value=RecordingAction("sell_from_starport", Result.fail("stop"), []),
        ) as sell_action:
            result = fsm.run(context)

        self.assertFalse(result.success)
        sell_action.assert_called_once()

    def test_trade_fsm_halts_sell_when_buy_price_is_unknown(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=15000)),
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=1,
        )
        fsm.allow_one_unverified_sell = False

        with patch("trade.SellFromStarport") as sell_action:
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertIn("source buy price is unknown", result.reason)
        sell_action.assert_not_called()

    def test_trade_fsm_allows_one_unverified_sell_when_starting_at_destination_sell(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=15000)),
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
        self.assertFalse(fsm.allow_one_unverified_sell)
        self.assertEqual(call_log[0], "sell_from_starport")

    def test_trade_fsm_keeps_one_unverified_sell_available_until_success(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = TradeFsm(
            source_station_name="Khun Port",
            commodity_name="Steel",
            destination_name="[BKRN] Event Horizon X5W-54J",
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=15000)),
            destination_is_carrier=True,
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=2,
        )

        sell_action = FlakyRecordingAction(
            "sell_from_starport",
            [Result.fail("station mismatch"), Result.ok("Sold cargo.")],
            call_log,
        )

        with (
            patch("trade.SellFromStarport", return_value=sell_action),
            patch("trade.time.sleep", return_value=None),
            patch(
                "trade.LeaveStation",
                return_value=RecordingAction("leave_station", Result.fail("stop"), call_log),
            ),
        ):
            result = fsm.run(context)

        self.assertFalse(result.success)
        self.assertEqual(call_log[:2], ["sell_from_starport", "sell_from_starport"])
        self.assertFalse(fsm.allow_one_unverified_sell)

    def test_trade_fsm_uses_shorter_source_leave_wait_for_carrier(self) -> None:
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. LAKE KASANE BZZ-NTG",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=FakeMarketDataSource(),
            source_is_carrier=True,
            cycle_limit=1,
            initial_state=TradeState.LEAVE_SOURCE,
            retry_attempts_per_state=1,
        )

        captured_kwargs: dict[str, object] = {}

        def build_leave_action(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return RecordingAction("leave_station", Result.fail("stop"), [])

        with patch("trade.LeaveStation", side_effect=build_leave_action):
            result = fsm.run(context)

        self.assertFalse(result.success)
        timings = captured_kwargs["timings"]
        self.assertEqual(timings.auto_launch_wait_seconds, DEFAULT_SOURCE_CONFIG["auto_launch_wait_seconds"])

    def test_trade_fsm_runs_expected_action_order(self) -> None:
        call_log: list[str] = []
        context = build_context()
        fsm = TradeFsm(
            source_station_name="P.T.N. BLUE TRADER",
            commodity_name="AGRONOMIC TREATMENT",
            destination_name="MILLE GATEWAY",
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=18000, demand=5000)),
            cycle_limit=1,
            retry_attempts_per_state=1,
        )

        with (
            patch(
                "trade.BuyFromStarport",
                return_value=RecordingAction(
                    "buy_from_starport",
                    Result.ok("Bought cargo.", debug={"unit_buy_price": 12000}),
                    call_log,
                ),
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
            market_data_source=SnapshotMarketDataSource(build_market_snapshot(sell_price=18000, demand=5000)),
            cycle_limit=1,
            initial_state=TradeState.SELL_AT_DESTINATION,
            retry_attempts_per_state=1,
        )
        fsm.last_source_buy_price = 12000

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
