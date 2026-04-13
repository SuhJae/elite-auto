from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AppConfig
from app.domain.context import Context
from app.domain.protocols import MarketDataSource
from app.domain.result import Result
from app.state.market_reader import MarketReader

if TYPE_CHECKING:
    from app.domain.models import CommodityListing, MarketSnapshot


STANDALONE_STATION_NAME: str | None = None
STANDALONE_COMMODITY = "COPPER"
STANDALONE_CONFIG_PATH: str | None = None
STANDALONE_ACTION = "sell"
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_KEY_INTERVAL_SECONDS: float | None = None
STANDALONE_LIST_STEP_INTERVAL_SECONDS: float | None = None
STANDALONE_POST_SELL_CONFIRM_WAIT_SECONDS: float | None = None
STANDALONE_POST_BACK_WAIT_SECONDS: float | None = None
STANDALONE_POST_SELL_CARGO_REFRESH_WAIT_SECONDS: float | None = None


@dataclass(slots=True)
class StarportSellTimings:
    """Named timings for the standalone market-sell flow."""

    key_interval_seconds: float = 0.5
    list_step_interval_seconds: float = 0.1
    filter_hold_seconds: float = 3.0
    services_open_wait_seconds: float = 5.0
    market_open_wait_seconds: float = 5.0
    sell_hold_seconds: float = 10.0
    post_sell_confirm_wait_seconds: float = 3.0
    post_back_wait_seconds: float = 2.0
    post_sell_cargo_refresh_wait_seconds: float = 1.0


@dataclass(slots=True)
class SellFromStarport:
    """Sell commodities after landing using the station market UI."""

    commodity: str
    market_data_source: MarketDataSource
    station_name: str | None = None
    is_top: bool = False
    timings: StarportSellTimings = field(default_factory=StarportSellTimings)

    name = "sell_from_starport"

    def run(self, context: Context) -> Result:
        if context.ship_control is None or context.input_adapter is None:
            return Result.fail("Input control is not available in the current context.")

        state = context.state_reader.snapshot()
        if not state.is_docked:
            return Result.fail(
                "Cannot run starport sell routine while undocked.",
                debug={"state": state.to_debug_dict()},
            )

        commodity_name = _normalize_commodity_name(self.commodity)
        if not commodity_name:
            return Result.fail("A commodity name is required.", debug={"commodity": self.commodity})

        cargo_path = _resolve_elite_path(context.config.paths.cargo_file, "Cargo.json")
        cargo_inventory = load_cargo_inventory(cargo_path)
        cargo_units_before = cargo_inventory.get(commodity_name, 0)
        if cargo_units_before <= 0:
            return Result.fail(
                "Requested commodity is not present in cargo.",
                debug={"commodity": self.commodity, "cargo_path": str(cargo_path)},
            )

        context.logger.info(
            "Starting starport sell routine",
            extra={
                "station_name": self.station_name,
                "commodity": self.commodity,
                "is_top": self.is_top,
                "cargo_units_before": cargo_units_before,
            },
        )

        self._open_station_services_and_market(context)
        snapshot = self.market_data_source.snapshot(required=True)

        if self.station_name:
            station_match = _normalize_text(snapshot.station_name) == _normalize_text(self.station_name)
            if not station_match:
                return Result.fail(
                    "Opened market does not match the requested station.",
                    debug={
                        "expected_station": self.station_name,
                        "actual_station": snapshot.station_name,
                        "star_system": snapshot.star_system,
                        "market_id": snapshot.market_id,
                    },
                )

        visible_items = get_sell_screen_items(snapshot)
        visible_names = {
            _normalize_commodity_name(_goods_name(commodity))
            for commodity in visible_items
        }
        visible_names.update(
            _normalize_commodity_name(commodity.name_localised or "")
            for commodity in visible_items
            if commodity.name_localised
        )

        if not _commodity_exists_anywhere(snapshot, commodity_name):
            return Result.fail(
                "Requested commodity is not present in this market.",
                debug={
                    "station_name": snapshot.station_name,
                    "commodity": self.commodity,
                    "visible_items": [_goods_name(item) for item in visible_items],
                },
            )

        if commodity_name not in visible_names:
            return Result.fail(
                "Requested commodity exists in market data but is not currently sellable here.",
                debug={
                    "station_name": snapshot.station_name,
                    "commodity": self.commodity,
                    "visible_items": [_goods_name(item) for item in visible_items],
                },
            )

        self._sell_one_filtered_item(context)

        time.sleep(self.timings.post_sell_cargo_refresh_wait_seconds)
        cargo_inventory_after = load_cargo_inventory(cargo_path)
        cargo_units_after = cargo_inventory_after.get(commodity_name, 0)
        if cargo_units_after >= cargo_units_before:
            return Result.fail(
                "Sell did not reduce cargo; stopping for safety.",
                debug={
                    "station_name": snapshot.station_name,
                    "commodity": self.commodity,
                    "cargo_units_before": cargo_units_before,
                    "cargo_units_after": cargo_units_after,
                    "cargo_path": str(cargo_path),
                },
            )

        return Result.ok(
            "Starport sell routine completed.",
            debug={
                "station_name": snapshot.station_name,
                "sold": self.commodity,
                "requested": self.commodity,
                "cargo_units_before": cargo_units_before,
                "cargo_units_after": cargo_units_after,
            },
        )

    def _open_station_services_and_market(self, context: Context) -> None:
        ship_control = context.ship_control
        if ship_control is None:
            raise RuntimeError("Ship control is not available.")

        # Refuel + repair from station services before opening commodities.
        ship_control.ui_select("up")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("select")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("right")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("select")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("right")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("select")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("down")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("select")

        time.sleep(self.timings.services_open_wait_seconds)

        ship_control.ui_select("down")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("select")

        time.sleep(self.timings.market_open_wait_seconds)

    def _sell_one_filtered_item(self, context: Context) -> None:
        ship_control = context.ship_control
        input_adapter = context.input_adapter
        if ship_control is None or input_adapter is None:
            raise RuntimeError("Input control is not available.")

        # Switch from the default Buy tab to the Sell tab first.
        ship_control.ui_select("down")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("select")
        _pause(self.timings.key_interval_seconds)

        if not self.is_top:
            # Open the filter menu and isolate the current cargo item.
            ship_control.ui_select("down")
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("down")
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("select")
            _pause(self.timings.key_interval_seconds)

            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("select")
            _pause(self.timings.key_interval_seconds)

            # Leave the filter UI and focus the filtered goods list.
        ship_control.ui_select("right")
        _pause(self.timings.key_interval_seconds)
        input_adapter.press(context.config.controls.ui_select)
        _pause(self.timings.key_interval_seconds)

        # Open the first filtered result and push the quantity slider to max cargo.
        ship_control.ui_select("select")
        _pause(self.timings.key_interval_seconds)
        input_adapter.hold(context.config.controls.ui_right, self.timings.sell_hold_seconds)
        _pause(self.timings.key_interval_seconds)

        ship_control.ui_select("down")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("select")
        _pause(self.timings.post_sell_confirm_wait_seconds)

        if not self.is_top:
            # Re-open filters and reset them before leaving the screen.
            input_adapter.press(context.config.controls.ui_select)
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("right")
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("right")
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("select")
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("left")
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("left")
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("select")
            _pause(self.timings.key_interval_seconds)

        input_adapter.press(context.config.controls.ui_back)
        _pause(self.timings.post_back_wait_seconds)
        input_adapter.press(context.config.controls.ui_back)
        _pause(self.timings.post_back_wait_seconds)

        ship_control.ui_select("down")
        _pause(self.timings.key_interval_seconds)


def print_sell_screen(snapshot: "MarketSnapshot", cargo_inventory: dict[str, int] | None = None) -> None:
    print(render_sell_screen(snapshot, cargo_inventory=cargo_inventory))


def render_sell_screen(snapshot: "MarketSnapshot", cargo_inventory: dict[str, int] | None = None) -> str:
    visible_items = get_sell_screen_items(snapshot)
    cargo_inventory = cargo_inventory or {}

    cargo_width = max(
        [len("Cargo"), *[len(format_cargo_count(commodity, cargo_inventory)) for commodity in visible_items]],
        default=len("Cargo"),
    )
    goods_width = max([len("Goods"), *[len(_goods_name(commodity)) for commodity in visible_items]], default=len("Goods"))
    demand_width = max([len("Demand"), *[len(format_demand(commodity)) for commodity in visible_items]], default=len("Demand"))
    sell_width = max([len("Sell"), *[len(format_sell_price(commodity)) for commodity in visible_items]], default=len("Sell"))

    lines = [
        f"Station: {snapshot.station_name or 'Unknown'}",
        f"System: {snapshot.star_system or 'Unknown'}",
        f"{'Cargo':>{cargo_width}}  {'Goods':<{goods_width}}  {'Demand':>{demand_width}}  {'Sell':>{sell_width}}",
        f"{'-' * cargo_width}  {'-' * goods_width}  {'-' * demand_width}  {'-' * sell_width}",
    ]

    for category_name, items_iter in groupby(visible_items, key=_category_name):
        lines.append(category_name)
        for commodity in items_iter:
            lines.append(
                f"{format_cargo_count(commodity, cargo_inventory):>{cargo_width}}  "
                f"{_goods_name(commodity):<{goods_width}}  "
                f"{format_demand(commodity):>{demand_width}}  "
                f"{format_sell_price(commodity):>{sell_width}}"
            )

    if not visible_items:
        lines.append("(no sell-screen items)")

    return "\n".join(lines)


def get_sell_screen_items(snapshot: "MarketSnapshot") -> list["CommodityListing"]:
    visible_items = [
        commodity
        for commodity in _dedupe_commodities(snapshot.commodities)
        if appears_on_sell_screen(commodity)
    ]
    visible_items.sort(key=lambda commodity: (_category_name(commodity), _goods_name(commodity)))
    return visible_items


def appears_on_sell_screen(commodity: "CommodityListing") -> bool:
    if commodity.sell_price <= 0:
        return False
    if commodity.demand > 0:
        return True
    return commodity.demand == 0 and (commodity.consumer or commodity.producer)


def format_cargo_count(commodity: "CommodityListing", cargo_inventory: dict[str, int]) -> str:
    return str(cargo_count_for_commodity(commodity, cargo_inventory))


def format_demand(commodity: "CommodityListing") -> str:
    return f"{commodity.demand:,}"


def format_sell_price(commodity: "CommodityListing") -> str:
    return f"{commodity.sell_price:,}"


def cargo_count_for_commodity(commodity: "CommodityListing", cargo_inventory: dict[str, int]) -> int:
    normalized_names = {
        _normalize_commodity_name(commodity.name),
        _normalize_commodity_name(commodity.name_localised or ""),
        _normalize_commodity_name(_goods_name(commodity)),
    }
    normalized_names.discard("")
    return max((cargo_inventory.get(name, 0) for name in normalized_names), default=0)


def load_cargo_inventory(cargo_path: str | Path) -> dict[str, int]:
    payload = _read_json_object(Path(cargo_path))
    inventory = payload.get("Inventory", [])
    if not isinstance(inventory, list):
        return {}

    cargo_inventory: defaultdict[str, int] = defaultdict(int)
    for item in inventory:
        if not isinstance(item, dict):
            continue

        count = int(item.get("Count", 0) or 0)
        if count <= 0:
            continue

        normalized_names = {
            _normalize_commodity_name(_localize_symbol_name(item.get("Name"))),
            _normalize_commodity_name(str(item.get("Name_Localised", "") or "")),
        }
        normalized_names.discard("")
        for normalized_name in normalized_names:
            cargo_inventory[normalized_name] += count

    return dict(cargo_inventory)


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _localize_symbol_name(value: object) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip("$")
    if cleaned.endswith("_name;"):
        cleaned = cleaned[:-6]
    elif cleaned.endswith(";"):
        cleaned = cleaned[:-1]
    return cleaned


def _normalize_commodity_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def _category_name(commodity: "CommodityListing") -> str:
    return (commodity.category_localised or commodity.category or "Unknown").strip()


def _goods_name(commodity: "CommodityListing") -> str:
    return (commodity.name_localised or commodity.name).strip()


def _dedupe_commodities(commodities: list["CommodityListing"]) -> list["CommodityListing"]:
    deduped: list["CommodityListing"] = []
    seen_keys: set[tuple[object, str, str]] = set()

    for commodity in commodities:
        dedupe_key = (
            commodity.commodity_id,
            _normalize_commodity_name(_category_name(commodity)),
            _normalize_commodity_name(_goods_name(commodity)),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped.append(commodity)

    return deduped


def _iter_sell_screen_item_rows(visible_items: list["CommodityListing"]) -> list["CommodityListing"]:
    item_rows: list["CommodityListing"] = []
    for _, items_iter in groupby(visible_items, key=_category_name):
        item_rows.extend(items_iter)
    return item_rows


def _commodity_exists_anywhere(snapshot: "MarketSnapshot", commodity_name: str) -> bool:
    for commodity in snapshot.commodities:
        if _normalize_commodity_name(commodity.name) == commodity_name:
            return True
        if commodity.name_localised and _normalize_commodity_name(commodity.name_localised) == commodity_name:
            return True
    return False


def _pause(seconds: float) -> None:
    time.sleep(seconds)


def _resolve_elite_path(path: Path, filename: str | None, prefer_directory: bool = False) -> Path:
    if path.exists():
        return path

    windows_elite_root = Path("/mnt/c/Users")
    if not windows_elite_root.exists():
        return path

    candidates = sorted(
        windows_elite_root.glob("*/Saved Games/Frontier Developments/Elite Dangerous"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    for candidate_root in candidates:
        candidate_path = candidate_root if prefer_directory or filename is None else candidate_root / filename
        if candidate_path.exists():
            return candidate_path

    return path


def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    market_path = _resolve_elite_path(config.paths.market_file, "Market.json")
    cargo_path = _resolve_elite_path(config.paths.cargo_file, "Cargo.json")
    journal_dir = _resolve_elite_path(config.paths.journal_dir, None, prefer_directory=True)
    market_reader = MarketReader(market_path=market_path, journal_dir=journal_dir)
    snapshot = market_reader.snapshot(required=True)
    cargo_inventory = load_cargo_inventory(cargo_path)

    if STANDALONE_ACTION == "print":
        print_sell_screen(snapshot, cargo_inventory=cargo_inventory)
        return 0

    from app.actions.starport_buy import build_standalone_context

    context = build_standalone_context(config)
    action = SellFromStarport(
        station_name=STANDALONE_STATION_NAME,
        commodity=STANDALONE_COMMODITY,
        market_data_source=market_reader,
        timings=StarportSellTimings(
            key_interval_seconds=STANDALONE_KEY_INTERVAL_SECONDS if STANDALONE_KEY_INTERVAL_SECONDS is not None else 0.5,
            list_step_interval_seconds=(
                STANDALONE_LIST_STEP_INTERVAL_SECONDS
                if STANDALONE_LIST_STEP_INTERVAL_SECONDS is not None
                else 0.1
            ),
            post_sell_confirm_wait_seconds=(
                STANDALONE_POST_SELL_CONFIRM_WAIT_SECONDS
                if STANDALONE_POST_SELL_CONFIRM_WAIT_SECONDS is not None
                else 3.0
            ),
            post_back_wait_seconds=(
                STANDALONE_POST_BACK_WAIT_SECONDS
                if STANDALONE_POST_BACK_WAIT_SECONDS is not None
                else 2.0
            ),
            post_sell_cargo_refresh_wait_seconds=(
                STANDALONE_POST_SELL_CARGO_REFRESH_WAIT_SECONDS
                if STANDALONE_POST_SELL_CARGO_REFRESH_WAIT_SECONDS is not None
                else 1.0
            ),
        ),
    )

    if STANDALONE_ACTION != "sell":
        print(f"Unknown standalone starport_sell action: {STANDALONE_ACTION}")
        return 2

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(
            f"Warning: focusing game window. Starting starport_sell for {STANDALONE_COMMODITY} "
            f"in {STANDALONE_START_DELAY_SECONDS:.1f} seconds..."
        )
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    result = action.run(context)
    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
