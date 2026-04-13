from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from app.actions.starport_buy import BuyFromStarport, StarportBuyTimings, render_buy_screen
from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import CommodityListing, DockedStationContext, MarketSnapshot, ShipState


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
    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        return None

    def key_down(self, key: str) -> None:
        return None

    def key_up(self, key: str) -> None:
        return None

    def hold(self, key: str, seconds: float) -> None:
        return None


class FakeShipControl:
    def set_throttle_percent(self, percent: int) -> None:
        return None

    def boost(self) -> None:
        return None

    def open_left_panel(self) -> None:
        return None

    def cycle_next_panel(self) -> None:
        return None

    def ui_select(self, direction: str = "select") -> None:
        return None

    def charge_fsd(self, target: str = "supercruise") -> None:
        return None


class RecordingShipControl(FakeShipControl):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ui_select(self, direction: str = "select") -> None:
        self.calls.append(direction)


class FakeMarketDataSource:
    def __init__(self, snapshot: MarketSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self, required: bool = False) -> MarketSnapshot:
        return self._snapshot


def build_context(is_docked: bool) -> Context:
    logger = logging.getLogger("elite_auto.test_starport_buy")
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
        input_adapter=FakeInputAdapter(),
        ship_control=FakeShipControl(),
    )


def build_context_with_ship_control(is_docked: bool, ship_control: FakeShipControl) -> Context:
    logger = logging.getLogger("elite_auto.test_starport_buy")
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
        input_adapter=FakeInputAdapter(),
        ship_control=ship_control,
    )


def build_snapshot(buy_price: int = 1200, stock: int = 42) -> MarketSnapshot:
    return MarketSnapshot(
        station_name="Jameson Memorial",
        station_type="Coriolis",
        star_system="Shinrarta Dezhra",
        market_id=128666762,
        timestamp="2026-04-07T23:30:00Z",
        commodities=[
            CommodityListing(
                commodity_id=1,
                name="metaalloys",
                name_localised="Meta-Alloys",
                category="industrial",
                category_localised="Industrial",
                buy_price=buy_price,
                sell_price=0,
                mean_price=0,
                stock=stock,
                demand=0,
                stock_bracket=0,
                demand_bracket=0,
                consumer=False,
                producer=False,
                rare=False,
            )
        ],
        source_path=Path(tempfile.gettempdir()) / "Market.json",
        source_timestamp=None,
        docked_context=DockedStationContext(
            station_name="Jameson Memorial",
            star_system="Shinrarta Dezhra",
            market_id=128666762,
            docked_timestamp="2026-04-07T23:20:00Z",
            market_timestamp="2026-04-07T23:30:00Z",
            journal_path=Path(tempfile.gettempdir()) / "Journal.log",
        ),
    )


class TestStarportBuy(unittest.TestCase):
    @staticmethod
    def _fast_timings() -> StarportBuyTimings:
        return StarportBuyTimings(
            key_interval_seconds=0.0,
            list_step_interval_seconds=0.0,
            services_open_wait_seconds=0.0,
            market_open_wait_seconds=0.0,
            buy_hold_seconds=0.0,
            post_buy_confirm_wait_seconds=0.0,
            post_back_wait_seconds=0.0,
        )

    def test_buy_from_starport_preflight_succeeds_for_matching_market(self) -> None:
        action = BuyFromStarport(
            station_name="Jameson Memorial",
            commodity="Meta-Alloys",
            market_data_source=FakeMarketDataSource(build_snapshot()),
            timings=self._fast_timings(),
        )

        result = action.run(build_context(is_docked=True))

        self.assertTrue(result.success)
        assert result.debug is not None
        self.assertEqual(result.debug["station_name"], "Jameson Memorial")

    def test_buy_from_starport_rejects_missing_commodity(self) -> None:
        action = BuyFromStarport(
            station_name="Jameson Memorial",
            commodity="Tritium",
            market_data_source=FakeMarketDataSource(build_snapshot()),
            timings=self._fast_timings(),
        )

        result = action.run(build_context(is_docked=True))

        self.assertFalse(result.success)
        self.assertIn("not present", result.reason.lower())

    def test_buy_from_starport_rejects_unbuyable_commodity(self) -> None:
        action = BuyFromStarport(
            station_name="Jameson Memorial",
            commodity="Meta-Alloys",
            market_data_source=FakeMarketDataSource(build_snapshot(buy_price=0, stock=0)),
            timings=self._fast_timings(),
        )

        result = action.run(build_context(is_docked=True))

        self.assertFalse(result.success)
        self.assertIn("not currently buyable", result.reason.lower())

    def test_buy_from_starport_requires_docked_state(self) -> None:
        action = BuyFromStarport(
            station_name="Jameson Memorial",
            commodity="Meta-Alloys",
            market_data_source=FakeMarketDataSource(build_snapshot()),
            timings=self._fast_timings(),
        )

        result = action.run(build_context(is_docked=False))

        self.assertFalse(result.success)
        self.assertIn("undocked", result.reason.lower())

    def test_buy_from_starport_uses_carrier_market_path_when_enabled(self) -> None:
        ship_control = RecordingShipControl()
        action = BuyFromStarport(
            station_name="Jameson Memorial",
            commodity="Meta-Alloys",
            market_data_source=FakeMarketDataSource(build_snapshot()),
            is_carrier=True,
            timings=self._fast_timings(),
        )

        result = action.run(build_context_with_ship_control(is_docked=True, ship_control=ship_control))

        self.assertTrue(result.success)
        self.assertEqual(
            ship_control.calls,
            [
                "select",
                "right",
                "right",
                "select",
                "right",
                "select",
                "down",
                "select",
                "down",
            ],
        )

    def test_buy_from_starport_allows_station_mismatch_when_price_is_within_cap(self) -> None:
        snapshot = build_snapshot(buy_price=15000, stock=42)
        snapshot.station_name = "BLX-91W"
        action = BuyFromStarport(
            station_name="BZZ-NTG",
            commodity="Meta-Alloys",
            market_data_source=FakeMarketDataSource(snapshot),
            max_buy_price=15500,
            timings=self._fast_timings(),
        )

        result = action.run(build_context(is_docked=True))

        self.assertTrue(result.success)

    def test_buy_from_starport_rejects_station_mismatch_when_price_exceeds_cap(self) -> None:
        snapshot = build_snapshot(buy_price=16000, stock=42)
        snapshot.station_name = "BLX-91W"
        action = BuyFromStarport(
            station_name="BZZ-NTG",
            commodity="Meta-Alloys",
            market_data_source=FakeMarketDataSource(snapshot),
            max_buy_price=15500,
            timings=self._fast_timings(),
        )

        result = action.run(build_context(is_docked=True))

        self.assertFalse(result.success)
        self.assertIn("does not match", result.reason.lower())

    def test_render_buy_screen_groups_and_sorts_categories_and_items(self) -> None:
        snapshot = MarketSnapshot(
            station_name="Test Station",
            station_type="Coriolis",
            star_system="Test System",
            market_id=1,
            timestamp="2026-04-07T23:30:00Z",
            commodities=[
                CommodityListing(
                    commodity_id=1,
                    name="silver",
                    name_localised="Silver",
                    category="metals",
                    category_localised="Metals",
                    buy_price=100,
                    sell_price=0,
                    mean_price=0,
                    stock=25,
                    demand=0,
                    stock_bracket=1,
                    demand_bracket=0,
                    consumer=False,
                    producer=True,
                    rare=False,
                ),
                CommodityListing(
                    commodity_id=2,
                    name="animalmeat",
                    name_localised="Animal Meat",
                    category="foods",
                    category_localised="Foods",
                    buy_price=50,
                    sell_price=0,
                    mean_price=0,
                    stock=10,
                    demand=0,
                    stock_bracket=1,
                    demand_bracket=0,
                    consumer=False,
                    producer=True,
                    rare=False,
                ),
                CommodityListing(
                    commodity_id=3,
                    name="gold",
                    name_localised="Gold",
                    category="metals",
                    category_localised="Metals",
                    buy_price=200,
                    sell_price=0,
                    mean_price=0,
                    stock=5,
                    demand=0,
                    stock_bracket=1,
                    demand_bracket=0,
                    consumer=False,
                    producer=True,
                    rare=False,
                ),
                CommodityListing(
                    commodity_id=4,
                    name="haematite",
                    name_localised="Haematite",
                    category="minerals",
                    category_localised="Minerals",
                    buy_price=0,
                    sell_price=1000,
                    mean_price=0,
                    stock=0,
                    demand=100,
                    stock_bracket=0,
                    demand_bracket=3,
                    consumer=True,
                    producer=False,
                    rare=False,
                ),
            ],
            source_path=Path(tempfile.gettempdir()) / "Market.json",
            source_timestamp=None,
            docked_context=None,
        )

        rendered = render_buy_screen(snapshot)

        foods_index = rendered.index("Foods")
        metals_index = rendered.index("Metals")
        animal_meat_index = rendered.index("Animal Meat")
        gold_index = rendered.index("Gold")
        silver_index = rendered.index("Silver")

        self.assertLess(foods_index, metals_index)
        self.assertLess(animal_meat_index, gold_index)
        self.assertLess(gold_index, silver_index)
        self.assertNotIn("Haematite", rendered)

    def test_render_buy_screen_excludes_zero_supply_rows(self) -> None:
        snapshot = MarketSnapshot(
            station_name="Test Station",
            station_type="Coriolis",
            star_system="Test System",
            market_id=1,
            timestamp="2026-04-07T23:30:00Z",
            commodities=[
                CommodityListing(
                    commodity_id=1,
                    name="zeroitem",
                    name_localised="Zero Item",
                    category="foods",
                    category_localised="Foods",
                    buy_price=100,
                    sell_price=0,
                    mean_price=0,
                    stock=0,
                    demand=0,
                    stock_bracket=0,
                    demand_bracket=0,
                    consumer=False,
                    producer=True,
                    rare=False,
                ),
                CommodityListing(
                    commodity_id=2,
                    name="realitem",
                    name_localised="Real Item",
                    category="foods",
                    category_localised="Foods",
                    buy_price=100,
                    sell_price=0,
                    mean_price=0,
                    stock=7,
                    demand=0,
                    stock_bracket=1,
                    demand_bracket=0,
                    consumer=False,
                    producer=True,
                    rare=False,
                ),
            ],
            source_path=Path(tempfile.gettempdir()) / "Market.json",
            source_timestamp=None,
            docked_context=None,
        )

        rendered = render_buy_screen(snapshot)

        self.assertIn("Real Item", rendered)
        self.assertNotIn("Zero Item", rendered)


if __name__ == "__main__":
    unittest.main()
