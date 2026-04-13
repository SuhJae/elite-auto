from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.actions.starport_sell import (
    SellFromStarport,
    StarportSellTimings,
    cargo_count_for_commodity,
    get_sell_screen_items,
    load_cargo_inventory,
    render_sell_screen,
)
from app.config import AppConfig
from app.domain.context import Context
from app.domain.models import CommodityListing, MarketSnapshot, ShipState


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


class RecordingInputAdapter:
    def __init__(self, call_log: list[tuple[str, str, float | int]]) -> None:
        self._call_log = call_log

    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        self._call_log.append(("press", key, presses))

    def key_down(self, key: str) -> None:
        self._call_log.append(("key_down", key, 0))

    def key_up(self, key: str) -> None:
        self._call_log.append(("key_up", key, 0))

    def hold(self, key: str, seconds: float) -> None:
        self._call_log.append(("hold", key, seconds))


class RecordingShipControl:
    def __init__(self, call_log: list[tuple[str, str, float | int]]) -> None:
        self._call_log = call_log

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
        self._call_log.append(("ui_select", direction, 0))

    def charge_fsd(self, target: str = "supercruise") -> None:
        return None


class FakeMarketDataSource:
    def __init__(self, snapshot: MarketSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self, required: bool = False) -> MarketSnapshot:
        return self._snapshot


def build_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        station_name="Moxon's Mojo",
        station_type="Orbis",
        star_system="Bolg",
        market_id=128000000,
        timestamp="2026-04-09T22:47:47Z",
        commodities=[
            CommodityListing(
                commodity_id=1,
                name="hydrogenperoxide",
                name_localised="Hydrogen Peroxide",
                category="chemicals",
                category_localised="Chemicals",
                buy_price=0,
                sell_price=3016,
                mean_price=0,
                stock=0,
                demand=4695,
                stock_bracket=0,
                demand_bracket=3,
                consumer=True,
                producer=False,
                rare=False,
            ),
            CommodityListing(
                commodity_id=2,
                name="water",
                name_localised="Water",
                category="chemicals",
                category_localised="Chemicals",
                buy_price=0,
                sell_price=606,
                mean_price=0,
                stock=0,
                demand=3136,
                stock_bracket=0,
                demand_bracket=3,
                consumer=True,
                producer=False,
                rare=False,
            ),
            CommodityListing(
                commodity_id=3,
                name="agronomictreatment",
                name_localised="Agronomic Treatment",
                category="chemicals",
                category_localised="Chemicals",
                buy_price=9519,
                sell_price=9284,
                mean_price=0,
                stock=362,
                demand=1,
                stock_bracket=1,
                demand_bracket=1,
                consumer=False,
                producer=True,
                rare=False,
            ),
            CommodityListing(
                commodity_id=4,
                name="clothing",
                name_localised="Clothing",
                category="consumeritems",
                category_localised="Consumer Items",
                buy_price=0,
                sell_price=1206,
                mean_price=0,
                stock=0,
                demand=5688,
                stock_bracket=0,
                demand_bracket=3,
                consumer=True,
                producer=False,
                rare=False,
            ),
            CommodityListing(
                commodity_id=5,
                name="copper",
                name_localised="Copper",
                category="metals",
                category_localised="Metals",
                buy_price=0,
                sell_price=620,
                mean_price=0,
                stock=0,
                demand=5500,
                stock_bracket=0,
                demand_bracket=3,
                consumer=True,
                producer=False,
                rare=False,
            ),
            CommodityListing(
                commodity_id=6,
                name="deltaphoenicispalms",
                name_localised="Delta Phoenicis Palms",
                category="chemicals",
                category_localised="Chemicals",
                buy_price=708,
                sell_price=707,
                mean_price=0,
                stock=0,
                demand=0,
                stock_bracket=0,
                demand_bracket=0,
                consumer=False,
                producer=False,
                rare=False,
            ),
        ],
        source_path=Path(tempfile.gettempdir()) / "Market.json",
        source_timestamp=None,
        docked_context=None,
    )


def build_context(cargo_path: Path, call_log: list[tuple[str, str, float | int]] | None = None) -> Context:
    logger = logging.getLogger("elite_auto.test_starport_sell")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    state = ShipState(
        is_docked=True,
        is_mass_locked=False,
        is_supercruise=False,
        cargo_count=0,
        gui_focus=None,
        status_flags=0,
    )
    config = AppConfig.default()
    config.paths.cargo_file = cargo_path
    call_log = call_log if call_log is not None else []
    return Context(
        config=config,
        logger=logger,
        debug_snapshot_dir=Path("debug_snapshots"),
        state_reader=FakeStateReader(state),
        event_stream=FakeEventStream(),
        input_adapter=RecordingInputAdapter(call_log),
        ship_control=RecordingShipControl(call_log),
    )


class TestStarportSell(unittest.TestCase):
    @staticmethod
    def _fast_timings() -> StarportSellTimings:
        return StarportSellTimings(
            key_interval_seconds=0.0,
            list_step_interval_seconds=0.0,
            filter_hold_seconds=0.0,
            services_open_wait_seconds=0.0,
            market_open_wait_seconds=0.0,
            sell_hold_seconds=0.0,
            post_sell_confirm_wait_seconds=0.0,
            post_back_wait_seconds=0.0,
            post_sell_cargo_refresh_wait_seconds=0.0,
        )

    def test_get_sell_screen_items_filters_to_station_buy_list(self) -> None:
        snapshot = build_snapshot()

        visible = get_sell_screen_items(snapshot)

        self.assertEqual(
            [item.name_localised for item in visible],
            ["Agronomic Treatment", "Hydrogen Peroxide", "Water", "Clothing", "Copper"],
        )

    def test_get_sell_screen_items_treats_zero_demand_as_infinite(self) -> None:
        snapshot = build_snapshot()
        snapshot.commodities[2].demand = 0

        visible = get_sell_screen_items(snapshot)

        self.assertIn("Agronomic Treatment", [item.name_localised for item in visible])

    def test_render_sell_screen_includes_cargo_column(self) -> None:
        snapshot = build_snapshot()

        rendered = render_sell_screen(
            snapshot,
            cargo_inventory={
                "hydrogenperoxide": 4,
                "clothing": 2,
            },
        )

        self.assertIn("Cargo", rendered)
        self.assertIn("Hydrogen Peroxide", rendered)
        self.assertIn("Clothing", rendered)
        self.assertIn("4", rendered)
        self.assertIn("2", rendered)
        self.assertIn("Agronomic Treatment", rendered)
        self.assertNotIn("Delta Phoenicis Palms", rendered)

    def test_cargo_count_for_commodity_matches_symbol_or_localized_name(self) -> None:
        snapshot = build_snapshot()
        commodity = snapshot.commodities[0]

        self.assertEqual(cargo_count_for_commodity(commodity, {"hydrogenperoxide": 3}), 3)
        self.assertEqual(cargo_count_for_commodity(commodity, {}), 0)

    def test_load_cargo_inventory_tracks_symbol_and_localized_names(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            cargo_path = Path(root) / "Cargo.json"
            cargo_path.write_text(
                """
                {
                  "Inventory": [
                    {"Name": "$hydrogenperoxide_name;", "Name_Localised": "Hydrogen Peroxide", "Count": 3},
                    {"Name": "clothing", "Name_Localised": "Clothing", "Count": 2}
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            cargo_inventory = load_cargo_inventory(cargo_path)

        self.assertEqual(cargo_inventory["hydrogenperoxide"], 3)
        self.assertEqual(cargo_inventory["clothing"], 2)

    def test_render_sell_screen_dedupes_repeated_rows_before_grouping(self) -> None:
        snapshot = build_snapshot()
        snapshot.commodities.append(snapshot.commodities[3])

        rendered = render_sell_screen(snapshot)

        self.assertEqual(rendered.count("Consumer Items"), 1)
        self.assertEqual(rendered.count("Clothing"), 1)

    def test_sell_from_starport_rejects_missing_cargo(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            cargo_path = Path(root) / "Cargo.json"
            cargo_path.write_text('{"Inventory": []}', encoding="utf-8")
            context = build_context(cargo_path)
            action = SellFromStarport(
                station_name="Moxon's Mojo",
                commodity="Copper",
                market_data_source=FakeMarketDataSource(build_snapshot()),
                timings=self._fast_timings(),
            )

            result = action.run(context)

        self.assertFalse(result.success)
        self.assertIn("not present in cargo", result.reason.lower())

    def test_sell_from_starport_uses_sell_tab_then_sells_selected_row(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            cargo_path = Path(root) / "Cargo.json"
            cargo_path.write_text(
                """
                {
                  "Inventory": [
                    {"Name": "$copper_name;", "Name_Localised": "Copper", "Count": 12}
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )
            call_log: list[tuple[str, str, float | int]] = []
            context = build_context(cargo_path, call_log=call_log)
            action = SellFromStarport(
                station_name="Moxon's Mojo",
                commodity="Copper",
                market_data_source=FakeMarketDataSource(build_snapshot()),
                timings=self._fast_timings(),
            )

            with patch(
                "app.actions.starport_sell.load_cargo_inventory",
                side_effect=[
                    {"copper": 12},
                    {"copper": 0},
                ],
            ):
                result = action.run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            call_log,
            [
                ("ui_select", "up", 0),
                ("ui_select", "select", 0),
                ("ui_select", "right", 0),
                ("ui_select", "select", 0),
                ("ui_select", "right", 0),
                ("ui_select", "select", 0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("ui_select", "down", 0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("ui_select", "select", 0),
                ("ui_select", "right", 0),
                ("press", "space", 1),
                ("ui_select", "select", 0),
                ("hold", "right", 0.0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("press", "space", 1),
                ("ui_select", "right", 0),
                ("ui_select", "right", 0),
                ("ui_select", "select", 0),
                ("ui_select", "left", 0),
                ("ui_select", "left", 0),
                ("ui_select", "select", 0),
                ("press", "backspace", 1),
                ("press", "backspace", 1),
                ("ui_select", "down", 0),
            ],
        )

    def test_sell_from_starport_skips_filter_flow_when_item_is_top(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            cargo_path = Path(root) / "Cargo.json"
            cargo_path.write_text(
                """
                {
                  "Inventory": [
                    {"Name": "$copper_name;", "Name_Localised": "Copper", "Count": 12}
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )
            call_log: list[tuple[str, str, float | int]] = []
            context = build_context(cargo_path, call_log=call_log)
            action = SellFromStarport(
                station_name="Moxon's Mojo",
                commodity="Copper",
                is_top=True,
                market_data_source=FakeMarketDataSource(build_snapshot()),
                timings=self._fast_timings(),
            )

            with patch(
                "app.actions.starport_sell.load_cargo_inventory",
                side_effect=[
                    {"copper": 12},
                    {"copper": 0},
                ],
            ):
                result = action.run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            call_log,
            [
                ("ui_select", "up", 0),
                ("ui_select", "select", 0),
                ("ui_select", "right", 0),
                ("ui_select", "select", 0),
                ("ui_select", "right", 0),
                ("ui_select", "select", 0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("ui_select", "right", 0),
                ("press", "space", 1),
                ("ui_select", "select", 0),
                ("hold", "right", 0.0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("press", "backspace", 1),
                ("press", "backspace", 1),
                ("ui_select", "down", 0),
            ],
        )

    def test_sell_from_starport_uses_carrier_market_path_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            cargo_path = Path(root) / "Cargo.json"
            cargo_path.write_text(
                """
                {
                  "Inventory": [
                    {"Name": "$copper_name;", "Name_Localised": "Copper", "Count": 12}
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )
            call_log: list[tuple[str, str, float | int]] = []
            context = build_context(cargo_path, call_log=call_log)
            action = SellFromStarport(
                station_name="Moxon's Mojo",
                commodity="Copper",
                is_carrier=True,
                is_top=True,
                market_data_source=FakeMarketDataSource(build_snapshot()),
                timings=self._fast_timings(),
            )

            with patch(
                "app.actions.starport_sell.load_cargo_inventory",
                side_effect=[
                    {"copper": 12},
                    {"copper": 0},
                ],
            ):
                result = action.run(context)

        self.assertTrue(result.success)
        self.assertEqual(
            call_log,
            [
                ("ui_select", "select", 0),
                ("ui_select", "right", 0),
                ("ui_select", "right", 0),
                ("ui_select", "select", 0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("ui_select", "right", 0),
                ("press", "space", 1),
                ("ui_select", "select", 0),
                ("hold", "right", 0.0),
                ("ui_select", "down", 0),
                ("ui_select", "select", 0),
                ("press", "backspace", 1),
                ("press", "backspace", 1),
                ("ui_select", "down", 0),
            ],
        )

    def test_sell_from_starport_fails_safe_if_cargo_does_not_decrease(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            cargo_path = Path(root) / "Cargo.json"
            cargo_path.write_text(
                """
                {
                  "Inventory": [
                    {"Name": "$copper_name;", "Name_Localised": "Copper", "Count": 12}
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )
            call_log: list[tuple[str, str, float | int]] = []
            context = build_context(cargo_path, call_log=call_log)
            action = SellFromStarport(
                station_name="Moxon's Mojo",
                commodity="Copper",
                market_data_source=FakeMarketDataSource(build_snapshot()),
                timings=self._fast_timings(),
            )

            with patch(
                "app.actions.starport_sell.load_cargo_inventory",
                side_effect=[
                    {"copper": 12},
                    {"copper": 12},
                ],
            ):
                result = action.run(context)

        self.assertFalse(result.success)
        self.assertIn("did not reduce cargo", result.reason.lower())


if __name__ == "__main__":
    unittest.main()
