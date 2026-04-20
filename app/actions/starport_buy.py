from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.adapters.capture_dxcam import DxcamCapture
from app.adapters.input_pydirect import PydirectInputAdapter, PydirectInputShipControl, ShipKeyMap
from app.adapters.vision_cv import OpenCVVisionSystem
from app.config import AppConfig, configure_logging
from app.domain.context import Context
from app.domain.protocols import MarketDataSource
from app.domain.result import Result
from app.actions.starport_sell import _resolve_elite_path, load_cargo_inventory
from app.state.bindings_reader import EliteBindingsReader
from app.state.cargo_reader import CargoReader
from app.state.journal_tailer import JournalTailer
from app.state.market_reader import MarketReader
from app.state.status_reader import EliteStateReader, StatusFileReader

if TYPE_CHECKING:
    from app.domain.models import CommodityListing, MarketSnapshot


# Edit these values for standalone testing of this file.
STANDALONE_STATION_NAME = "Hildebrandt Refinery"
STANDALONE_COMMODITY = "SCRAP"
STANDALONE_CONFIG_PATH: str | None = None
STANDALONE_PRINT_BUY_SCREEN = False
STANDALONE_DEBUG_LOG_KEYS = True
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_KEY_INTERVAL_SECONDS: float | None = None
STANDALONE_PRESS_HOLD_SECONDS: float | None = None
STANDALONE_LIST_STEP_INTERVAL_SECONDS: float | None = None
STANDALONE_POST_BUY_CONFIRM_WAIT_SECONDS: float | None = None
STANDALONE_POST_BACK_WAIT_SECONDS: float | None = None


@dataclass(slots=True)
class StarportBuyTimings:
    """Named timings for the standalone market-buy flow."""

    key_interval_seconds: float = 0.5
    list_step_interval_seconds: float = 0.1
    services_open_wait_seconds: float = 5.0
    market_open_wait_seconds: float = 5.0
    buy_hold_seconds: float = 10.0
    post_buy_confirm_wait_seconds: float = 3.0
    post_back_wait_seconds: float = 2.0


@dataclass(slots=True)
class BuyFromStarport:
    """Buy commodities after landing using a simple one-file UI routine."""

    station_name: str
    commodity: str
    market_data_source: MarketDataSource
    is_carrier: bool = False
    max_buy_price: int | None = None
    timings: StarportBuyTimings = field(default_factory=StarportBuyTimings)

    name = "buy_from_starport"

    def run(self, context: Context) -> Result:
        if context.ship_control is None or context.input_adapter is None:
            return Result.fail("Input control is not available in the current context.")

        state = context.state_reader.snapshot()
        if not state.is_docked:
            return Result.fail(
                "Cannot run starport buy routine while undocked.",
                debug={"state": state.to_debug_dict()},
            )

        context.logger.info(
            "Starting starport buy routine",
            extra={
                "station_name": self.station_name,
                "commodity": self.commodity,
                "is_carrier": self.is_carrier,
                "max_buy_price": self.max_buy_price,
            },
        )

        commodity_name = _normalize_commodity_name(self.commodity)
        if not commodity_name:
            return Result.fail("A commodity name is required.", debug={"commodity": self.commodity})

        cargo_path = _resolve_elite_path(context.config.paths.cargo_file, "Cargo.json")
        cargo_inventory_before = load_cargo_inventory(cargo_path)
        cargo_units_before = cargo_inventory_before.get(commodity_name, 0)

        self._open_station_services_and_market(context)
        snapshot = self.market_data_source.snapshot(required=True)

        visible_items = get_buy_screen_items(snapshot)
        matching_item = next(
            (
                commodity
                for commodity in visible_items
                if _normalize_commodity_name(_goods_name(commodity)) == commodity_name
                or _normalize_commodity_name(commodity.name_localised or "") == commodity_name
            ),
            None,
        )

        station_match = _normalize_text(snapshot.station_name) == _normalize_text(self.station_name)
        if not station_match:
            if (
                self.max_buy_price is not None
                and matching_item is not None
                and matching_item.buy_price <= self.max_buy_price
            ):
                context.logger.warning(
                    "Opened market station name did not match, but proceeding because commodity price is within cap.",
                    extra={
                        "expected_station": self.station_name,
                        "actual_station": snapshot.station_name,
                        "commodity": self.commodity,
                        "buy_price": matching_item.buy_price,
                        "max_buy_price": self.max_buy_price,
                        "star_system": snapshot.star_system,
                        "market_id": snapshot.market_id,
                    },
                )
            else:
                return Result.fail(
                    "Opened market does not match the requested station.",
                    debug={
                        "expected_station": self.station_name,
                        "actual_station": snapshot.station_name,
                        "star_system": snapshot.star_system,
                        "market_id": snapshot.market_id,
                        "commodity": self.commodity,
                        "matched_buy_price": matching_item.buy_price if matching_item is not None else None,
                        "max_buy_price": self.max_buy_price,
                    },
                )

        commodity_positions = {
            _normalize_commodity_name(_goods_name(commodity)): index
            for index, commodity in enumerate(visible_items, start=1)
        }
        localized_positions = {
            _normalize_commodity_name(commodity.name_localised or ""): index
            for index, commodity in enumerate(visible_items, start=1)
            if commodity.name_localised
        }
        commodity_positions.update(localized_positions)

        list_index = commodity_positions.get(commodity_name)
        if list_index is None and not _commodity_exists_anywhere(snapshot, commodity_name):
            return Result.fail(
                "Requested commodity is not present in this market.",
                debug={
                    "station_name": snapshot.station_name,
                    "commodity": self.commodity,
                    "visible_items": [_goods_name(item) for item in visible_items],
                },
            )

        if list_index is None:
            return Result.fail(
                "Requested commodity exists in market data but is not currently buyable.",
                debug={
                    "station_name": snapshot.station_name,
                    "commodity": self.commodity,
                },
            )

        if not visible_items:
            return Result.fail("No buyable commodities are visible in the current market.", debug=snapshot.to_dict())

        self._buy_one_visible_item(context, list_index)

        cargo_inventory_after = load_cargo_inventory(cargo_path)
        cargo_units_after = cargo_inventory_after.get(commodity_name, 0)
        cargo_units_bought = cargo_units_after - cargo_units_before
        if cargo_units_bought <= 0:
            return Result.fail(
                "Buy did not increase cargo; stopping for safety.",
                debug={
                    "station_name": snapshot.station_name,
                    "commodity": self.commodity,
                    "cargo_units_before": cargo_units_before,
                    "cargo_units_after": cargo_units_after,
                    "cargo_path": str(cargo_path),
                },
            )

        return Result.ok(
            "Starport buy routine completed.",
            debug={
                "station_name": snapshot.station_name,
                "purchased": self.commodity,
                "requested": self.commodity,
                "max_buy_price": self.max_buy_price,
                "unit_buy_price": matching_item.buy_price if matching_item is not None else None,
                "supply_before": matching_item.stock if matching_item is not None else None,
                "cargo_units_before": cargo_units_before,
                "cargo_units_after": cargo_units_after,
                "cargo_units_bought": cargo_units_bought,
            },
        )

    def _open_station_services_and_market(self, context: Context) -> None:
        ship_control = context.ship_control
        if ship_control is None:
            raise RuntimeError("Ship control is not available.")

        # Refuel + repair from station services.
        if self.is_carrier is False:
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

        if self.is_carrier:
            ship_control.ui_select("right")
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("right")
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("select")
        else:
            ship_control.ui_select("down")
            _pause(self.timings.key_interval_seconds)
            ship_control.ui_select("select")

        time.sleep(self.timings.market_open_wait_seconds)

    def _buy_one_visible_item(self, context: Context, list_index: int) -> None:
        ship_control = context.ship_control
        input_adapter = context.input_adapter
        if ship_control is None or input_adapter is None:
            raise RuntimeError("Input control is not available.")

        ship_control.ui_select("right")
        _pause(self.timings.key_interval_seconds)

        for _ in range(max(0, list_index - 1)):
            ship_control.ui_select("down")
            _pause(self.timings.list_step_interval_seconds)

        ship_control.ui_select("select")
        _pause(self.timings.key_interval_seconds * 4)

        input_adapter.hold(context.config.controls.ui_right, self.timings.buy_hold_seconds)
        _pause(self.timings.key_interval_seconds)

        ship_control.ui_select("down")
        _pause(self.timings.key_interval_seconds)
        ship_control.ui_select("select")
        _pause(self.timings.post_buy_confirm_wait_seconds)

        input_adapter.press(context.config.controls.ui_back)
        _pause(self.timings.post_back_wait_seconds)
        input_adapter.press(context.config.controls.ui_back)
        _pause(self.timings.post_back_wait_seconds)


def print_buy_screen(snapshot: "MarketSnapshot") -> None:
    print(render_buy_screen(snapshot))


def render_buy_screen(snapshot: "MarketSnapshot") -> str:
    visible_items = get_buy_screen_items(snapshot)
    goods_width = max([len("Goods"), *[len(_goods_name(item)) for item in visible_items]], default=len("Goods"))
    supply_width = max([len("Supply"), *[len(format_supply(item)) for item in visible_items]], default=len("Supply"))
    buy_width = max([len("Buy"), *[len(format_buy_price(item)) for item in visible_items]], default=len("Buy"))

    lines = [
        f"Station: {snapshot.station_name or 'Unknown'}",
        f"System: {snapshot.star_system or 'Unknown'}",
        f"{'Goods':<{goods_width}}  {'Supply':>{supply_width}}  {'Buy':>{buy_width}}",
        f"{'-' * goods_width}  {'-' * supply_width}  {'-' * buy_width}",
    ]

    for category_name, items_iter in groupby(visible_items, key=_category_name):
        lines.append(category_name)
        for commodity in items_iter:
            lines.append(
                f"{_goods_name(commodity):<{goods_width}}  {format_supply(commodity):>{supply_width}}  {format_buy_price(commodity):>{buy_width}}"
            )

    if not visible_items:
        lines.append("(no buy-screen items)")

    return "\n".join(lines)


def get_buy_screen_items(snapshot: "MarketSnapshot") -> list["CommodityListing"]:
    visible_items = [commodity for commodity in snapshot.commodities if appears_on_buy_screen(commodity)]
    visible_items.sort(key=lambda commodity: (_category_name(commodity), _goods_name(commodity)))
    return visible_items


def appears_on_buy_screen(commodity: "CommodityListing") -> bool:
    return commodity.buy_price > 0 and commodity.stock > 0


def format_supply(commodity: "CommodityListing") -> str:
    return str(commodity.stock)


def format_buy_price(commodity: "CommodityListing") -> str:
    return str(commodity.buy_price)


def _commodity_exists_anywhere(snapshot: "MarketSnapshot", commodity_name: str) -> bool:
    for commodity in snapshot.commodities:
        if _normalize_commodity_name(commodity.name) == commodity_name:
            return True
        if commodity.name_localised and _normalize_commodity_name(commodity.name_localised) == commodity_name:
            return True
    return False


def _normalize_commodity_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def _category_name(commodity: "CommodityListing") -> str:
    return (commodity.category_localised or commodity.category or "Unknown").strip()


def _goods_name(commodity: "CommodityListing") -> str:
    return (commodity.name_localised or commodity.name).strip()


def _pause(seconds: float) -> None:
    time.sleep(seconds)


class DebugInputAdapter:
    """Optional debug wrapper that logs every low-level key action."""

    def __init__(self, wrapped: PydirectInputAdapter, logger) -> None:
        self._wrapped = wrapped
        self._logger = logger

    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        self._logger.info("Input press", extra={"key": key, "presses": presses, "interval": interval})
        self._wrapped.press(key, presses=presses, interval=interval)

    def key_down(self, key: str) -> None:
        self._logger.info("Input key_down", extra={"key": key})
        self._wrapped.key_down(key)

    def key_up(self, key: str) -> None:
        self._logger.info("Input key_up", extra={"key": key})
        self._wrapped.key_up(key)

    def hold(self, key: str, seconds: float) -> None:
        self._logger.info("Input hold", extra={"key": key, "seconds": seconds})
        self._wrapped.hold(key, seconds)


class DebugShipControl:
    """Optional debug wrapper that logs semantic ship-control calls."""

    def __init__(self, wrapped: PydirectInputShipControl, logger) -> None:
        self._wrapped = wrapped
        self._logger = logger

    def set_throttle_percent(self, percent: int) -> None:
        self._logger.info("ShipControl set_throttle_percent", extra={"percent": percent})
        self._wrapped.set_throttle_percent(percent)

    def boost(self) -> None:
        self._logger.info("ShipControl boost")
        self._wrapped.boost()

    def open_left_panel(self) -> None:
        self._logger.info("ShipControl open_left_panel")
        self._wrapped.open_left_panel()

    def cycle_previous_panel(self) -> None:
        self._logger.info("ShipControl cycle_previous_panel")
        self._wrapped.cycle_previous_panel()

    def cycle_next_panel(self) -> None:
        self._logger.info("ShipControl cycle_next_panel")
        self._wrapped.cycle_next_panel()

    def ui_select(self, direction: str = "select") -> None:
        self._logger.info("ShipControl ui_select", extra={"direction": direction})
        self._wrapped.ui_select(direction)

    def charge_fsd(self, target: str = "supercruise") -> None:
        self._logger.info("ShipControl charge_fsd", extra={"target": target})
        self._wrapped.charge_fsd(target)


def build_standalone_context(config: AppConfig) -> Context:
    logger = configure_logging(config, logger_name="elite_auto.starport_buy")
    cargo_reader = CargoReader(config.paths.cargo_file)
    state_reader = EliteStateReader(
        status_reader=StatusFileReader(config.paths.status_file),
        cargo_reader=cargo_reader,
        logger=logger,
    )
    event_stream = JournalTailer(config.paths.journal_dir, logger=logger)
    detected_bindings = EliteBindingsReader(config.paths, logger=logger).detect_controls(config.controls)
    controls = detected_bindings.controls if detected_bindings is not None else config.controls
    config.controls = controls

    keymap = ShipKeyMap(
        throttle_zero=controls.throttle_zero,
        throttle_fifty=controls.throttle_fifty,
        throttle_seventy_five=controls.throttle_seventy_five,
        throttle_full=controls.throttle_full,
        throttle_reverse_full=controls.throttle_reverse_full,
        boost=controls.boost,
        open_left_panel=controls.open_left_panel,
        cycle_previous_panel=controls.cycle_previous_panel,
        cycle_next_panel=controls.cycle_next_panel,
        ui_up=controls.ui_up,
        ui_down=controls.ui_down,
        ui_left=controls.ui_left,
        ui_right=controls.ui_right,
        ui_select=controls.ui_select,
        charge_fsd=controls.charge_fsd,
    )
    base_input_adapter = PydirectInputAdapter(
        press_hold_seconds=(
            STANDALONE_PRESS_HOLD_SECONDS
            if STANDALONE_PRESS_HOLD_SECONDS is not None
            else 0.12
        )
    )
    base_ship_control = PydirectInputShipControl(base_input_adapter, keymap)

    if STANDALONE_DEBUG_LOG_KEYS:
        input_adapter = DebugInputAdapter(base_input_adapter, logger)
        ship_control = DebugShipControl(base_ship_control, logger)
    else:
        input_adapter = base_input_adapter
        ship_control = base_ship_control

    return Context(
        config=config,
        logger=logger,
        debug_snapshot_dir=config.paths.debug_snapshots_dir,
        state_reader=state_reader,
        event_stream=event_stream,
        input_adapter=input_adapter,
        ship_control=ship_control,
        capture=DxcamCapture(),
        vision=OpenCVVisionSystem(config.paths.debug_snapshots_dir),
    )


def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    context = build_standalone_context(config)
    market_reader = MarketReader(
        market_path=config.paths.market_file,
        journal_dir=config.paths.journal_dir,
        logger=context.logger,
    )

    if STANDALONE_PRINT_BUY_SCREEN:
        print_buy_screen(market_reader.snapshot(required=True))
        return 0

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(
            f"Warning: focusing game window. Starting starport_buy in {STANDALONE_START_DELAY_SECONDS:.1f} seconds..."
        )
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    action = BuyFromStarport(
        station_name=STANDALONE_STATION_NAME,
        commodity=STANDALONE_COMMODITY,
        market_data_source=market_reader,
        timings=StarportBuyTimings(
            key_interval_seconds=(
                STANDALONE_KEY_INTERVAL_SECONDS
                if STANDALONE_KEY_INTERVAL_SECONDS is not None
                else 0.5
            ),
            list_step_interval_seconds=(
                STANDALONE_LIST_STEP_INTERVAL_SECONDS
                if STANDALONE_LIST_STEP_INTERVAL_SECONDS is not None
                else 0.1
            ),
            post_buy_confirm_wait_seconds=(
                STANDALONE_POST_BUY_CONFIRM_WAIT_SECONDS
                if STANDALONE_POST_BUY_CONFIRM_WAIT_SECONDS is not None
                else 3.0
            ),
            post_back_wait_seconds=(
                STANDALONE_POST_BACK_WAIT_SECONDS
                if STANDALONE_POST_BACK_WAIT_SECONDS is not None
                else 2.0
            ),
        ),
    )
    result = action.run(context)
    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
