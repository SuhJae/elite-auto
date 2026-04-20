from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.actions.align import AlignConfig, AlignToTargetCompass
from app.actions.docking import DockingTimings, RequestDockingSequence
from app.actions.fsd import EngageFsdSequence, FsdTimings
from app.actions.leave_station import LeaveStation, LeaveStationTimings
from app.actions.navigation import LockNavDestination, NavigationTimings
from app.actions.starport_buy import BuyFromStarport, StarportBuyTimings
from app.actions.starport_sell import SellFromStarport, StarportSellTimings
from app.config import AppConfig, configure_logging
from app.domain.context import Context
from app.domain.models import CommodityListing
from app.domain.protocols import MarketDataSource
from app.domain.result import Result
from app.state.market_reader import MarketReader
from run import build_context

DEFAULT_TRADE_CONFIG_PATH = Path(__file__).with_name("trade.config.json")
REQUIRED_TRADE_CONFIG_SECTIONS = ("sources", "destinations", "completion_station", "safety", "runtime")
REQUIRED_SOURCE_KEYS = (
    "id",
    "station_name",
    "commodity_name",
    "market_name",
    "is_carrier",
    "auto_launch_wait_seconds",
    "starting_units",
)
REQUIRED_DESTINATION_KEYS = (
    "id",
    "station_name",
    "commodity_name",
    "market_name",
    "is_carrier",
    "item_is_top",
    "auto_launch_wait_seconds",
    "starting_units",
)
REQUIRED_COMPLETION_KEYS = ("station_name", "is_carrier")
REQUIRED_SAFETY_KEYS = ("source_max_buy_price", "min_profit_per_unit")
REQUIRED_RUNTIME_KEYS = (
    "cargo_capacity_units",
    "start_delay_seconds",
    "retry_attempts_per_state",
    "retry_initial_wait_seconds",
    "retry_max_wait_seconds",
)


def _load_trade_config(path: str | Path | None) -> dict[str, object]:
    config_path = Path(path or DEFAULT_TRADE_CONFIG_PATH).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Trade config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Trade config file must contain a JSON object: {config_path}")

    missing_sections = [section for section in REQUIRED_TRADE_CONFIG_SECTIONS if section not in raw]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Trade config is missing required sections: {missing}")

    _validate_config_list(raw["sources"], REQUIRED_SOURCE_KEYS, "sources", config_path)
    _validate_config_list(raw["destinations"], REQUIRED_DESTINATION_KEYS, "destinations", config_path)
    _validate_config_object(raw["completion_station"], REQUIRED_COMPLETION_KEYS, "completion_station", config_path)
    _validate_config_object(raw["safety"], REQUIRED_SAFETY_KEYS, "safety", config_path)
    _validate_config_object(raw["runtime"], REQUIRED_RUNTIME_KEYS, "runtime", config_path)
    return raw


def _validate_config_list(
    value: object,
    required_keys: tuple[str, ...],
    section_name: str,
    config_path: Path,
) -> None:
    if not isinstance(value, list):
        raise ValueError(f"Trade config section '{section_name}' must be a list in {config_path}")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Trade config section '{section_name}[{index}]' must be an object in {config_path}")
        missing_keys = [key for key in required_keys if key not in item]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(f"Trade config section '{section_name}[{index}]' is missing required keys: {missing}")


def _validate_config_object(
    value: object,
    required_keys: tuple[str, ...],
    section_name: str,
    config_path: Path,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Trade config section '{section_name}' must be an object in {config_path}")
    missing_keys = [key for key in required_keys if key not in value]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Trade config section '{section_name}' is missing required keys: {missing}")


def _default_market_name(station_name: str, is_carrier: bool) -> str:
    if not is_carrier:
        return station_name

    tokens = station_name.strip().split()
    if not tokens:
        return station_name
    return tokens[-1]


def _normalize_commodity_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _normalize_station_identity(value: str | None) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


@dataclass(slots=True)
class TradeSource:
    id: str
    station_name: str
    commodity_name: str
    market_name: str | None
    is_carrier: bool
    auto_launch_wait_seconds: float
    starting_units: int | None
    remaining_units: int | None = None
    done: bool = False

    def reset(self, cargo_capacity_units: int) -> None:
        self.remaining_units = self.starting_units
        self.done = self.remaining_units is not None and self.remaining_units < cargo_capacity_units

    def to_config(self) -> dict[str, object]:
        return {
            "id": self.id,
            "station_name": self.station_name,
            "commodity_name": self.commodity_name,
            "market_name": self.market_name,
            "is_carrier": self.is_carrier,
            "auto_launch_wait_seconds": self.auto_launch_wait_seconds,
            "starting_units": self.starting_units,
            "remaining_units": self.remaining_units,
            "done": self.done,
        }


@dataclass(slots=True)
class TradeDestination:
    id: str
    station_name: str
    commodity_name: str
    market_name: str | None
    is_carrier: bool
    item_is_top: bool
    auto_launch_wait_seconds: float
    starting_units: int | None
    remaining_units: int | None = None
    done: bool = False

    def reset(self, cargo_capacity_units: int) -> None:
        self.remaining_units = self.starting_units
        self.done = self.remaining_units is not None and self.remaining_units < cargo_capacity_units

    def to_config(self) -> dict[str, object]:
        return {
            "id": self.id,
            "station_name": self.station_name,
            "commodity_name": self.commodity_name,
            "market_name": self.market_name,
            "is_carrier": self.is_carrier,
            "item_is_top": self.item_is_top,
            "auto_launch_wait_seconds": self.auto_launch_wait_seconds,
            "starting_units": self.starting_units,
            "remaining_units": self.remaining_units,
            "done": self.done,
        }


@dataclass(slots=True)
class CompletionStation:
    station_name: str
    is_carrier: bool
    market_name: str | None = None

    def to_config(self) -> dict[str, object]:
        return {
            "station_name": self.station_name,
            "market_name": self.market_name,
            "is_carrier": self.is_carrier,
        }


class TradeState(str, Enum):
    BUY_FROM_SOURCE = "buy_from_source"
    LEAVE_SOURCE = "leave_source"
    LOCK_NAV_TO_DESTINATION = "lock_nav_to_destination"
    ALIGN_TO_DESTINATION = "align_to_destination"
    ENGAGE_FSD_TO_DESTINATION = "engage_fsd_to_destination"
    DOCK_AT_DESTINATION = "dock_at_destination"
    SELL_AT_DESTINATION = "sell_at_destination"
    LEAVE_DESTINATION = "leave_destination"
    LOCK_NAV_TO_SOURCE = "lock_nav_to_source"
    ALIGN_TO_SOURCE = "align_to_source"
    ENGAGE_FSD_TO_SOURCE = "engage_fsd_to_source"
    DOCK_AT_SOURCE = "dock_at_source"
    LOCK_NAV_TO_COMPLETION = "lock_nav_to_completion"
    ALIGN_TO_COMPLETION = "align_to_completion"
    ENGAGE_FSD_TO_COMPLETION = "engage_fsd_to_completion"
    DOCK_AT_COMPLETION = "dock_at_completion"
    COMPLETED = "completed"


STAGE_ALIASES: dict[str, TradeState] = {
    "buy": TradeState.BUY_FROM_SOURCE,
    "buy-source": TradeState.BUY_FROM_SOURCE,
    "leave-source": TradeState.LEAVE_SOURCE,
    "nav-destination": TradeState.LOCK_NAV_TO_DESTINATION,
    "lock-nav-destination": TradeState.LOCK_NAV_TO_DESTINATION,
    "align-destination": TradeState.ALIGN_TO_DESTINATION,
    "fsd-destination": TradeState.ENGAGE_FSD_TO_DESTINATION,
    "dock-destination": TradeState.DOCK_AT_DESTINATION,
    "sell": TradeState.SELL_AT_DESTINATION,
    "sell-destination": TradeState.SELL_AT_DESTINATION,
    "leave-destination": TradeState.LEAVE_DESTINATION,
    "nav-source": TradeState.LOCK_NAV_TO_SOURCE,
    "lock-nav-source": TradeState.LOCK_NAV_TO_SOURCE,
    "align-source": TradeState.ALIGN_TO_SOURCE,
    "fsd-source": TradeState.ENGAGE_FSD_TO_SOURCE,
    "dock-source": TradeState.DOCK_AT_SOURCE,
    "nav-completion": TradeState.LOCK_NAV_TO_COMPLETION,
    "lock-nav-completion": TradeState.LOCK_NAV_TO_COMPLETION,
    "align-completion": TradeState.ALIGN_TO_COMPLETION,
    "fsd-completion": TradeState.ENGAGE_FSD_TO_COMPLETION,
    "dock-completion": TradeState.DOCK_AT_COMPLETION,
}


@dataclass(slots=True)
class TradeFsm:
    source_station_name: str
    commodity_name: str
    destination_name: str
    market_data_source: MarketDataSource
    source_is_carrier: bool = False
    destination_is_carrier: bool = True
    source_market_name: str | None = None
    destination_market_name: str | None = None
    source_max_buy_price: int | None = 7000
    destination_item_is_top: bool = True
    min_profit_per_unit: int = 5000
    destination_min_demand: int | None = None
    buy_timings: StarportBuyTimings = field(default_factory=StarportBuyTimings)
    sell_timings: StarportSellTimings = field(default_factory=StarportSellTimings)
    leave_station_timings: LeaveStationTimings = field(default_factory=LeaveStationTimings)
    navigation_timings: NavigationTimings = field(default_factory=NavigationTimings)
    align_config: AlignConfig = field(
        default_factory=lambda: AlignConfig(
            debug_window_enabled=False,
            debug_snapshot_interval_seconds=0.0,
            final_reticle_enabled=True,
            near_center_consensus_pause_seconds=2.0,
            near_center_consensus_samples=3,
            near_center_consensus_span_seconds=0.30,
        )
    )
    pre_dock_align_config: AlignConfig = field(
        default_factory=lambda: AlignConfig(
            alignment_tolerance_px=4.0,
            axis_alignment_tolerance_px=4.0,
            debug_window_enabled=False,
            debug_snapshot_interval_seconds=0.0,
            final_reticle_enabled=True,
            near_center_consensus_pause_seconds=2.0,
            near_center_consensus_samples=3,
            near_center_consensus_span_seconds=0.30,
        )
    )
    fsd_timings: FsdTimings = field(default_factory=FsdTimings)
    docking_timings: DockingTimings = field(default_factory=DockingTimings)
    cycle_limit: int | None = None
    start_delay_seconds: float = 3.0
    retry_attempts_per_state: int = 5
    retry_initial_wait_seconds: float = 10.0
    retry_max_wait_seconds: float = 300.0
    source_auto_launch_wait_seconds: float = 60.0
    destination_auto_launch_wait_seconds: float = 45.0
    cargo_capacity_units: int = 1154
    sources: list[TradeSource] = field(default_factory=list)
    destinations: list[TradeDestination] = field(default_factory=list)
    completion_station: CompletionStation | None = None
    trade_config_path: Path | None = None
    initial_state: TradeState = TradeState.BUY_FROM_SOURCE
    active_source_id: str | None = None
    active_destination_id: str | None = None
    last_source_buy_price: int | None = field(default=None, init=False)
    allow_one_unverified_sell: bool = field(default=False, init=False)
    completion_requested: bool = field(default=False, init=False)

    name = "trade"

    def __post_init__(self) -> None:
        if not self.sources:
            self.sources = [
                TradeSource(
                    id="source-1",
                    station_name=self.source_station_name,
                    commodity_name=self.commodity_name,
                    market_name=self.source_market_name,
                    is_carrier=self.source_is_carrier,
                    auto_launch_wait_seconds=self.source_auto_launch_wait_seconds,
                    starting_units=None,
                )
            ]
        if not self.destinations:
            self.destinations = [
                TradeDestination(
                    id="destination-1",
                    station_name=self.destination_name,
                    commodity_name=self.commodity_name,
                    market_name=self.destination_market_name,
                    is_carrier=self.destination_is_carrier,
                    item_is_top=self.destination_item_is_top,
                    auto_launch_wait_seconds=self.destination_auto_launch_wait_seconds,
                    starting_units=None,
                )
            ]
        for source in self.sources:
            if source.remaining_units is None and source.starting_units is not None:
                source.reset(self.cargo_capacity_units)
        for destination in self.destinations:
            if destination.remaining_units is None and destination.starting_units is not None:
                destination.reset(self.cargo_capacity_units)
        self.allow_one_unverified_sell = self.initial_state is TradeState.SELL_AT_DESTINATION
        self._ensure_active_route()

    def source_config(self) -> dict[str, object]:
        current_source = self._current_source()
        return {
            "station_name": self.source_station_name,
            "commodity_name": self.commodity_name,
            "market_name": self.source_market_name,
            "is_carrier": self.source_is_carrier,
            "auto_launch_wait_seconds": self.source_auto_launch_wait_seconds,
            "remaining_units": current_source.remaining_units if current_source is not None else None,
            "source_id": current_source.id if current_source is not None else None,
        }

    def destination_config(self) -> dict[str, object]:
        current_destination = self._current_destination()
        return {
            "station_name": self.destination_name,
            "market_name": self.destination_market_name,
            "is_carrier": self.destination_is_carrier,
            "item_is_top": self.destination_item_is_top,
            "auto_launch_wait_seconds": self.destination_auto_launch_wait_seconds,
            "remaining_units": current_destination.remaining_units if current_destination is not None else None,
            "destination_id": current_destination.id if current_destination is not None else None,
        }

    def safety_config(self) -> dict[str, object]:
        return {
            "source_max_buy_price": self.source_max_buy_price,
            "min_profit_per_unit": self.min_profit_per_unit,
            "last_source_buy_price": self.last_source_buy_price,
            "allow_one_unverified_sell": self.allow_one_unverified_sell,
        }

    def runtime_config(self) -> dict[str, object]:
        return {
            "cargo_capacity_units": self.cargo_capacity_units,
            "start_delay_seconds": self.start_delay_seconds,
            "retry_attempts_per_state": self.retry_attempts_per_state,
            "retry_initial_wait_seconds": self.retry_initial_wait_seconds,
            "retry_max_wait_seconds": self.retry_max_wait_seconds,
            "active_source_id": self.active_source_id,
            "active_destination_id": self.active_destination_id,
            "completion_requested": self.completion_requested,
        }

    def _current_source(self) -> TradeSource | None:
        if self.active_source_id is None:
            return None
        return next((source for source in self.sources if source.id == self.active_source_id), None)

    def _current_destination(self) -> TradeDestination | None:
        if self.active_destination_id is None:
            return None
        return next((destination for destination in self.destinations if destination.id == self.active_destination_id), None)

    def _find_source_by_station_name(self, station_name: str | None) -> TradeSource | None:
        station_key = _normalize_station_identity(station_name)
        if not station_key:
            return None
        for source in self.sources:
            source_keys = {
                _normalize_station_identity(source.station_name),
                _normalize_station_identity(source.market_name),
                _normalize_station_identity(_default_market_name(source.station_name, source.is_carrier)),
            }
            source_keys.discard("")
            if station_key in source_keys:
                return source
        return None

    def _find_destination_by_station_name(self, station_name: str | None) -> TradeDestination | None:
        station_key = _normalize_station_identity(station_name)
        if not station_key:
            return None
        for destination in self.destinations:
            destination_keys = {
                _normalize_station_identity(destination.station_name),
                _normalize_station_identity(destination.market_name),
                _normalize_station_identity(_default_market_name(destination.station_name, destination.is_carrier)),
            }
            destination_keys.discard("")
            if station_key in destination_keys:
                return destination
        return None

    def _set_active_source(self, source: TradeSource) -> None:
        self.active_source_id = source.id
        self.source_station_name = source.station_name
        self.source_market_name = source.market_name or _default_market_name(source.station_name, source.is_carrier)
        self.source_is_carrier = source.is_carrier
        self.source_auto_launch_wait_seconds = source.auto_launch_wait_seconds
        self.commodity_name = source.commodity_name

    def _set_active_destination(self, destination: TradeDestination) -> None:
        self.active_destination_id = destination.id
        self.destination_name = destination.station_name
        self.destination_market_name = destination.market_name or _default_market_name(
            destination.station_name,
            destination.is_carrier,
        )
        self.destination_is_carrier = destination.is_carrier
        self.destination_item_is_top = destination.item_is_top
        self.destination_auto_launch_wait_seconds = destination.auto_launch_wait_seconds

    def _apply_route(self, source: TradeSource, destination: TradeDestination) -> None:
        self._set_active_source(source)
        self._set_active_destination(destination)
        self.completion_requested = False
        self._persist_trade_config()

    def _route_is_viable(self, source: TradeSource, destination: TradeDestination) -> bool:
        if source.done or destination.done:
            return False
        if _normalize_commodity_name(source.commodity_name) != _normalize_commodity_name(destination.commodity_name):
            return False
        if source.remaining_units is not None and source.remaining_units < self.cargo_capacity_units:
            return False
        if destination.remaining_units is not None and destination.remaining_units < self.cargo_capacity_units:
            return False
        return True

    def _ensure_active_route(self) -> bool:
        current_source = self._current_source()
        current_destination = self._current_destination()
        if current_source is not None and current_destination is not None and self._route_is_viable(current_source, current_destination):
            self._apply_route(current_source, current_destination)
            return True

        best_pair: tuple[TradeSource, TradeDestination] | None = None
        best_transferable = -1.0
        for source in self.sources:
            for destination in self.destinations:
                if not self._route_is_viable(source, destination):
                    continue
                source_units = float("inf") if source.remaining_units is None else float(source.remaining_units)
                destination_units = float("inf") if destination.remaining_units is None else float(destination.remaining_units)
                transferable = min(source_units, destination_units)
                if transferable > best_transferable:
                    best_transferable = transferable
                    best_pair = (source, destination)
        if best_pair is None:
            self.active_source_id = None
            self.active_destination_id = None
            self._persist_trade_config()
            return False

        self._apply_route(*best_pair)
        return True

    def _candidate_destination_names(self) -> list[str]:
        names: list[str] = []
        active_destination = self._current_destination()
        if active_destination is not None:
            names.append(active_destination.station_name)
        active_source = self._current_source()
        if active_source is None:
            return names
        source_commodity = _normalize_commodity_name(active_source.commodity_name)
        for destination in self.destinations:
            if destination.station_name in names:
                continue
            if destination.done:
                continue
            if _normalize_commodity_name(destination.commodity_name) != source_commodity:
                continue
            if destination.remaining_units is not None and destination.remaining_units < self.cargo_capacity_units:
                continue
            names.append(destination.station_name)
        return names

    def _candidate_source_names(self) -> list[str]:
        names: list[str] = []
        active_source = self._current_source()
        if active_source is not None:
            names.append(active_source.station_name)
        active_destination = self._current_destination()
        if active_destination is None:
            return names
        destination_commodity = _normalize_commodity_name(active_destination.commodity_name)
        for source in self.sources:
            if source.station_name in names:
                continue
            if source.done:
                continue
            if _normalize_commodity_name(source.commodity_name) != destination_commodity:
                continue
            if source.remaining_units is not None and source.remaining_units < self.cargo_capacity_units:
                continue
            names.append(source.station_name)
        return names

    def _mark_source_progress(self, units_bought: int, supply_before: int | None = None) -> None:
        source = self._current_source()
        if source is None:
            return
        if supply_before is not None:
            source.remaining_units = max(0, supply_before - units_bought)
        elif source.remaining_units is not None:
            source.remaining_units = max(0, source.remaining_units - units_bought)
        else:
            return
        source.done = source.remaining_units < self.cargo_capacity_units
        self._persist_trade_config()

    def _mark_destination_progress(self, units_sold: int, demand_before: int | None = None) -> None:
        destination = self._current_destination()
        if destination is None:
            return
        if demand_before is not None:
            destination.remaining_units = max(0, demand_before - units_sold)
        elif destination.remaining_units is not None:
            destination.remaining_units = max(0, destination.remaining_units - units_sold)
        else:
            return
        destination.done = destination.remaining_units < self.cargo_capacity_units
        self._persist_trade_config()

    def _queue_completion(self) -> bool:
        if self.completion_station is None:
            return False
        if not self.destination_is_carrier:
            self.completion_requested = False
            self._persist_trade_config()
            return False
        self.completion_requested = True
        self._persist_trade_config()
        return True

    def _persist_trade_config(self) -> None:
        if self.trade_config_path is None:
            return
        payload = {
            "sources": [source.to_config() for source in self.sources],
            "destinations": [destination.to_config() for destination in self.destinations],
            "completion_station": self.completion_station.to_config() if self.completion_station is not None else None,
            "safety": {
                "source_max_buy_price": self.source_max_buy_price,
                "min_profit_per_unit": self.min_profit_per_unit,
            },
            "runtime": {
                "cargo_capacity_units": self.cargo_capacity_units,
                "start_delay_seconds": self.start_delay_seconds,
                "retry_attempts_per_state": self.retry_attempts_per_state,
                "retry_initial_wait_seconds": self.retry_initial_wait_seconds,
                "retry_max_wait_seconds": self.retry_max_wait_seconds,
                "active_route": {
                    "source_id": self.active_source_id,
                    "destination_id": self.active_destination_id,
                    "commodity_name": self.commodity_name if self.active_source_id and self.active_destination_id else None,
                },
                "completion_requested": self.completion_requested,
            },
        }
        self.trade_config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def run(self, context: Context) -> Result:
        state = self.initial_state
        if self.active_source_id is None and state not in {
            TradeState.LOCK_NAV_TO_COMPLETION,
            TradeState.ALIGN_TO_COMPLETION,
            TradeState.ENGAGE_FSD_TO_COMPLETION,
            TradeState.DOCK_AT_COMPLETION,
        }:
            if self._queue_completion():
                state = TradeState.LOCK_NAV_TO_COMPLETION
            else:
                return Result.ok(
                    "No viable trade routes remain.",
                    debug={
                        "source": self.source_config(),
                        "destination": self.destination_config(),
                        "runtime": self.runtime_config(),
                    },
                )
        history: list[dict[str, object]] = []
        completed_cycles = 0
        context.logger.info(
            "Starting trade FSM",
            extra={
                "source": self.source_config(),
                "destination": self.destination_config(),
                "safety": self.safety_config(),
                "runtime": self.runtime_config(),
                "initial_state": state.value,
                "cycle_limit": self.cycle_limit,
            },
        )

        while state is not TradeState.COMPLETED:
            action = self._build_action(state)
            result, state_history = self._run_state_with_retries(context, state, action)
            history.extend(state_history)

            if not result.success:
                return Result.fail(
                    f"Trade FSM failed in state '{state.value}': {result.reason}",
                    debug={
                        "failed_state": state.value,
                        "history": history,
                        "step_debug": result.debug,
                        "last_attempt": state_history[-1] if state_history else None,
                        "source": self.source_config(),
                        "destination": self.destination_config(),
                        "safety": self.safety_config(),
                        "runtime": self.runtime_config(),
                        "completed_cycles": completed_cycles,
                    },
                )

            self._capture_state_output(state, result)
            next_state, completed_cycles = self._next_state(state, completed_cycles)
            context.logger.info(
                "Trade FSM transition",
                extra={
                    "from_state": state.value,
                    "to_state": next_state.value,
                    "completed_cycles": completed_cycles,
                },
            )
            state = next_state

        return Result.ok(
            "Trade FSM completed.",
            debug={
                "history": history,
                "source": self.source_config(),
                "destination": self.destination_config(),
                "safety": self.safety_config(),
                "runtime": self.runtime_config(),
                "completed_cycles": completed_cycles,
            },
        )

    def _run_state_with_retries(
        self,
        context: Context,
        state: TradeState,
        action: object,
    ) -> tuple[Result, list[dict[str, object]]]:
        total_attempts = max(1, self.retry_attempts_per_state)
        state_history: list[dict[str, object]] = []

        for attempt in range(1, total_attempts + 1):
            context.logger.info(
                "Trade FSM state start",
                extra={
                    "state": state.value,
                    "action": getattr(action, "name", type(action).__name__),
                    "attempt": attempt,
                    "max_attempts": total_attempts,
                },
            )
            try:
                result = action.run(context)
            except Exception as exc:
                context.logger.exception(
                    "Trade FSM state raised exception",
                    extra={
                        "state": state.value,
                        "action": getattr(action, "name", type(action).__name__),
                        "attempt": attempt,
                        "max_attempts": total_attempts,
                        "error": str(exc),
                    },
                )
                result = Result.fail(
                    f"Unhandled exception in state '{state.value}': {exc}",
                    debug={"exception": str(exc)},
                )

            history_item: dict[str, object] = {
                "state": state.value,
                "action": getattr(action, "name", type(action).__name__),
                "attempt": attempt,
                "success": result.success,
                "reason": result.reason,
            }
            if result.debug is not None:
                history_item["debug"] = result.debug
            state_history.append(history_item)

            if result.success:
                return result, state_history

            if attempt >= total_attempts:
                return result, state_history

            wait_seconds = self._retry_wait_seconds(attempt)
            context.logger.warning(
                "Trade FSM state failed; retrying",
                extra={
                    "state": state.value,
                    "action": getattr(action, "name", type(action).__name__),
                    "attempt": attempt,
                    "max_attempts": total_attempts,
                    "wait_seconds": wait_seconds,
                    "reason": result.reason,
                },
            )
            time.sleep(wait_seconds)

        return Result.fail(f"State '{state.value}' exhausted retries unexpectedly."), state_history

    def _retry_wait_seconds(self, failed_attempt_index: int) -> float:
        base = max(0.0, self.retry_initial_wait_seconds)
        ceiling = max(base, self.retry_max_wait_seconds)
        if failed_attempt_index <= 1:
            return min(base, ceiling)
        return min(base * math.pow(2.0, failed_attempt_index - 1), ceiling)

    def _capture_state_output(self, state: TradeState, result: Result) -> None:
        if result.debug is None:
            return
        if state is TradeState.BUY_FROM_SOURCE:
            actual_station_name = result.debug.get("station_name")
            if isinstance(actual_station_name, str):
                actual_source = self._find_source_by_station_name(actual_station_name)
                if actual_source is not None:
                    self._set_active_source(actual_source)
            unit_buy_price = result.debug.get("unit_buy_price")
            if isinstance(unit_buy_price, int):
                self.last_source_buy_price = unit_buy_price
            units_bought = result.debug.get("cargo_units_bought")
            if isinstance(units_bought, int) and units_bought > 0:
                supply_before = result.debug.get("supply_before")
                self._mark_source_progress(units_bought, supply_before if isinstance(supply_before, int) else None)
            return
        if state is TradeState.SELL_AT_DESTINATION:
            actual_station_name = result.debug.get("station_name")
            if isinstance(actual_station_name, str):
                actual_destination = self._find_destination_by_station_name(actual_station_name)
                if actual_destination is not None:
                    self._set_active_destination(actual_destination)
            units_sold = result.debug.get("cargo_units_sold")
            if isinstance(units_sold, int) and units_sold > 0:
                demand_before = result.debug.get("demand_before")
                self._mark_destination_progress(units_sold, demand_before if isinstance(demand_before, int) else None)

    def _find_snapshot_commodity(self, commodity_name: str, commodities: list[CommodityListing]) -> CommodityListing | None:
        normalized_name = _normalize_commodity_name(commodity_name)
        for commodity in commodities:
            if _normalize_commodity_name(commodity.name) == normalized_name:
                return commodity
            if commodity.name_localised and _normalize_commodity_name(commodity.name_localised) == normalized_name:
                return commodity
        return None

    def _validate_destination_market(self, snapshot) -> Result | None:
        matching_item = self._find_snapshot_commodity(self.commodity_name, snapshot.commodities)
        if matching_item is None:
            return Result.fail(
                "Cannot verify destination profit because the commodity is missing from the current market snapshot.",
                debug={
                    "commodity": self.commodity_name,
                    "destination_name": self.destination_name,
                    "station_name": snapshot.station_name,
                },
            )

        if matching_item.demand < self.cargo_capacity_units:
            return Result.fail(
                "Destination demand is below cargo capacity; halting before sell.",
                debug={
                    "commodity": self.commodity_name,
                    "destination_name": self.destination_name,
                    "station_name": snapshot.station_name,
                    "destination_demand": matching_item.demand,
                    "cargo_capacity_units": self.cargo_capacity_units,
                },
            )

        unit_profit = matching_item.sell_price - self.last_source_buy_price
        if unit_profit < self.min_profit_per_unit:
            return Result.fail(
                "Destination profit per unit is below the configured minimum; halting before sell.",
                debug={
                    "commodity": self.commodity_name,
                    "destination_name": self.destination_name,
                    "station_name": snapshot.station_name,
                    "source_buy_price": self.last_source_buy_price,
                    "destination_sell_price": matching_item.sell_price,
                    "unit_profit": unit_profit,
                    "min_profit_per_unit": self.min_profit_per_unit,
                },
            )

        return None

    def _sell_with_profit_guard(self, context: Context) -> Result:
        if self.last_source_buy_price is None:
            if self.allow_one_unverified_sell:
                context.logger.warning(
                    "Skipping destination profit verification once because this run resumed at sell without a known source buy price.",
                    extra={
                        "commodity": self.commodity_name,
                        "destination_name": self.destination_name,
                    },
                )
                result = SellFromStarport(
                    station_name=self.destination_market_name or self.destination_name,
                    commodity=self.commodity_name,
                    market_data_source=self.market_data_source,
                    is_carrier=self.destination_is_carrier,
                    is_top=self.destination_item_is_top,
                    allow_station_mismatch_if_market_validation_passes=True,
                    timings=self.sell_timings,
                ).run(context)
                if result.success:
                    self.allow_one_unverified_sell = False
                return result
            return Result.fail(
                "Cannot verify destination profit because the source buy price is unknown.",
                debug={
                    "commodity": self.commodity_name,
                    "destination_name": self.destination_name,
                    "min_profit_per_unit": self.min_profit_per_unit,
                },
            )

        return SellFromStarport(
            station_name=self.destination_market_name or self.destination_name,
            commodity=self.commodity_name,
            market_data_source=self.market_data_source,
            is_carrier=self.destination_is_carrier,
            is_top=self.destination_item_is_top,
            market_validation=self._validate_destination_market,
            allow_station_mismatch_if_market_validation_passes=True,
            timings=self.sell_timings,
        ).run(context)

    def _leave_station_action(self, auto_launch_wait_seconds: float) -> LeaveStation:
        leave_timings = LeaveStationTimings(
            auto_launch_wait_seconds=auto_launch_wait_seconds,
            mass_lock_poll_interval_seconds=self.leave_station_timings.mass_lock_poll_interval_seconds,
            post_mass_lock_clear_wait_seconds=self.leave_station_timings.post_mass_lock_clear_wait_seconds,
            mass_lock_timeout_seconds=self.leave_station_timings.mass_lock_timeout_seconds,
        )
        return LeaveStation(timings=leave_timings)

    def _build_action(self, state: TradeState):
        if state is TradeState.BUY_FROM_SOURCE:
            return BuyFromStarport(
                station_name=self.source_market_name or self.source_station_name,
                commodity=self.commodity_name,
                market_data_source=self.market_data_source,
                is_carrier=self.source_is_carrier,
                max_buy_price=self.source_max_buy_price,
                timings=self.buy_timings,
            )
        if state is TradeState.LEAVE_SOURCE:
            return self._leave_station_action(self.source_auto_launch_wait_seconds)
        if state is TradeState.LEAVE_DESTINATION:
            return self._leave_station_action(self.destination_auto_launch_wait_seconds)
        if state is TradeState.LOCK_NAV_TO_DESTINATION:
            return LockNavDestination(
                target_name=self.destination_name,
                target_names=self._candidate_destination_names(),
                timings=self.navigation_timings,
            )
        if state is TradeState.ALIGN_TO_DESTINATION:
            return AlignToTargetCompass(config=self.align_config)
        if state is TradeState.ENGAGE_FSD_TO_DESTINATION:
            return EngageFsdSequence(timings=self.fsd_timings)
        if state is TradeState.DOCK_AT_DESTINATION:
            return RequestDockingSequence(timings=self.docking_timings)
        if state is TradeState.SELL_AT_DESTINATION:
            return _CallableAction("sell_from_starport", self._sell_with_profit_guard)
        if state is TradeState.LOCK_NAV_TO_SOURCE:
            return LockNavDestination(
                target_name=self.source_station_name,
                target_names=self._candidate_source_names(),
                timings=self.navigation_timings,
            )
        if state is TradeState.ALIGN_TO_SOURCE:
            return AlignToTargetCompass(config=self.align_config)
        if state is TradeState.ENGAGE_FSD_TO_SOURCE:
            return EngageFsdSequence(timings=self.fsd_timings)
        if state is TradeState.DOCK_AT_SOURCE:
            return RequestDockingSequence(timings=self.docking_timings)
        if state is TradeState.LOCK_NAV_TO_COMPLETION:
            if self.completion_station is None:
                raise ValueError("Completion station is not configured.")
            return LockNavDestination(
                target_name=self.completion_station.station_name,
                timings=self.navigation_timings,
            )
        if state is TradeState.ALIGN_TO_COMPLETION:
            return AlignToTargetCompass(config=self.align_config)
        if state is TradeState.ENGAGE_FSD_TO_COMPLETION:
            return EngageFsdSequence(timings=self.fsd_timings)
        if state is TradeState.DOCK_AT_COMPLETION:
            return RequestDockingSequence(timings=self.docking_timings)
        raise ValueError(f"Unsupported trade state: {state!r}")

    def _next_state(self, state: TradeState, completed_cycles: int) -> tuple[TradeState, int]:
        transitions = {
            TradeState.BUY_FROM_SOURCE: TradeState.LOCK_NAV_TO_DESTINATION,
            TradeState.LOCK_NAV_TO_DESTINATION: TradeState.LEAVE_SOURCE,
            TradeState.LEAVE_SOURCE: TradeState.ALIGN_TO_DESTINATION,
            TradeState.DOCK_AT_DESTINATION: TradeState.SELL_AT_DESTINATION,
            TradeState.LOCK_NAV_TO_SOURCE: TradeState.LEAVE_DESTINATION,
            TradeState.LEAVE_DESTINATION: TradeState.ALIGN_TO_SOURCE,
            TradeState.ALIGN_TO_DESTINATION: TradeState.ENGAGE_FSD_TO_DESTINATION,
            TradeState.ENGAGE_FSD_TO_DESTINATION: TradeState.DOCK_AT_DESTINATION,
            TradeState.ALIGN_TO_SOURCE: TradeState.ENGAGE_FSD_TO_SOURCE,
            TradeState.ENGAGE_FSD_TO_SOURCE: TradeState.DOCK_AT_SOURCE,
            TradeState.LOCK_NAV_TO_COMPLETION: TradeState.ALIGN_TO_COMPLETION,
            TradeState.ALIGN_TO_COMPLETION: TradeState.ENGAGE_FSD_TO_COMPLETION,
            TradeState.ENGAGE_FSD_TO_COMPLETION: TradeState.DOCK_AT_COMPLETION,
        }
        if state is TradeState.SELL_AT_DESTINATION:
            if self._ensure_active_route():
                return TradeState.LOCK_NAV_TO_SOURCE, completed_cycles
            if self._queue_completion():
                return TradeState.LOCK_NAV_TO_COMPLETION, completed_cycles
            return TradeState.COMPLETED, completed_cycles
        if state is TradeState.DOCK_AT_SOURCE:
            completed_cycles += 1
            if self.cycle_limit is not None and completed_cycles >= self.cycle_limit:
                return TradeState.COMPLETED, completed_cycles
            return TradeState.BUY_FROM_SOURCE, completed_cycles
        if state is TradeState.DOCK_AT_COMPLETION:
            return TradeState.COMPLETED, completed_cycles
        return transitions[state], completed_cycles


@dataclass(slots=True)
class _CallableAction:
    name: str
    runner: Callable[[Context], Result]

    def run(self, context: Context) -> Result:
        return self.runner(context)


def build_trade_fsm(
    config: AppConfig,
    trade_config: dict[str, object],
    trade_config_path: Path,
    logger: logging.Logger | None = None,
) -> TradeFsm:
    sources_raw = trade_config["sources"]
    destinations_raw = trade_config["destinations"]
    completion_raw = trade_config["completion_station"]
    safety = trade_config["safety"]
    runtime = trade_config["runtime"]
    if not isinstance(sources_raw, list) or not isinstance(destinations_raw, list):
        raise ValueError("Trade config sources and destinations must be lists.")
    if not isinstance(completion_raw, dict) or not isinstance(safety, dict) or not isinstance(runtime, dict):
        raise ValueError("Trade config completion_station, safety, and runtime must be objects.")

    sources = [
        TradeSource(
            id=str(item["id"]),
            station_name=str(item["station_name"]),
            commodity_name=str(item["commodity_name"]),
            market_name=item.get("market_name"),
            is_carrier=bool(item["is_carrier"]),
            auto_launch_wait_seconds=float(item["auto_launch_wait_seconds"]),
            starting_units=None if item["starting_units"] is None else int(item["starting_units"]),
        )
        for item in sources_raw
        if isinstance(item, dict)
    ]
    destinations = [
        TradeDestination(
            id=str(item["id"]),
            station_name=str(item["station_name"]),
            commodity_name=str(item["commodity_name"]),
            market_name=item.get("market_name"),
            is_carrier=bool(item["is_carrier"]),
            item_is_top=bool(item["item_is_top"]),
            auto_launch_wait_seconds=float(item["auto_launch_wait_seconds"]),
            starting_units=None if item["starting_units"] is None else int(item["starting_units"]),
        )
        for item in destinations_raw
        if isinstance(item, dict)
    ]
    completion_station = CompletionStation(
        station_name=str(completion_raw["station_name"]),
        market_name=completion_raw.get("market_name"),
        is_carrier=bool(completion_raw["is_carrier"]),
    )
    cargo_capacity_units = int(runtime["cargo_capacity_units"])
    for source in sources:
        source.reset(cargo_capacity_units)
    for destination in destinations:
        destination.reset(cargo_capacity_units)
    market_reader = MarketReader(
        market_path=config.paths.market_file,
        journal_dir=config.paths.journal_dir,
        logger=logger or configure_logging(config, logger_name="elite_auto.trade"),
    )
    initial_source = sources[0] if sources else None
    initial_destination = destinations[0] if destinations else None
    return TradeFsm(
        source_station_name=initial_source.station_name if initial_source is not None else completion_station.station_name,
        commodity_name=initial_source.commodity_name if initial_source is not None else "",
        destination_name=initial_destination.station_name if initial_destination is not None else completion_station.station_name,
        market_data_source=market_reader,
        source_is_carrier=bool(initial_source.is_carrier) if initial_source is not None else False,
        destination_is_carrier=bool(initial_destination.is_carrier) if initial_destination is not None else False,
        source_market_name=initial_source.market_name if initial_source is not None else None,
        destination_market_name=initial_destination.market_name if initial_destination is not None else None,
        source_max_buy_price=safety["source_max_buy_price"],
        destination_item_is_top=bool(initial_destination.item_is_top) if initial_destination is not None else False,
        min_profit_per_unit=int(safety["min_profit_per_unit"]),
        start_delay_seconds=float(runtime["start_delay_seconds"]),
        retry_attempts_per_state=int(runtime["retry_attempts_per_state"]),
        retry_initial_wait_seconds=float(runtime["retry_initial_wait_seconds"]),
        retry_max_wait_seconds=float(runtime["retry_max_wait_seconds"]),
        source_auto_launch_wait_seconds=float(initial_source.auto_launch_wait_seconds) if initial_source is not None else 60.0,
        destination_auto_launch_wait_seconds=(
            float(initial_destination.auto_launch_wait_seconds) if initial_destination is not None else 45.0
        ),
        cargo_capacity_units=cargo_capacity_units,
        sources=sources,
        destinations=destinations,
        completion_station=completion_station,
        trade_config_path=trade_config_path,
    )


def parse_start_stage(value: str) -> TradeState:
    normalized = value.strip().lower().replace("_", "-")
    alias_match = STAGE_ALIASES.get(normalized)
    if alias_match is not None:
        return alias_match

    try:
        return TradeState(normalized.replace("-", "_"))
    except ValueError as exc:
        available = ", ".join(sorted(STAGE_ALIASES))
        raise argparse.ArgumentTypeError(
            f"Unknown start stage '{value}'. Available aliases: {available}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trade FSM.")
    parser.add_argument("--config", help="Optional path to a JSON config file.")
    parser.add_argument(
        "--trade-config",
        default=str(DEFAULT_TRADE_CONFIG_PATH),
        help="Path to the trade-specific JSON config file.",
    )
    parser.add_argument(
        "--start-stage",
        type=parse_start_stage,
        help="Optional stage to resume from, for example sell-destination or dock-source.",
    )
    parser.add_argument(
        "--cycle-limit",
        type=int,
        help="Optional number of full source-to-source loops to run. Omit for infinite looping.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=None,
        help="Attempts per FSM state before quitting. Default: 5.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=None,
        help="Seconds to wait before starting so the game window can be focused.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.from_json(args.config) if args.config else AppConfig.default()
    trade_config_path = Path(args.trade_config).expanduser().resolve()
    trade_config = _load_trade_config(trade_config_path)
    context = build_context(config)
    runtime = trade_config["runtime"]
    safety = trade_config["safety"]
    if not isinstance(runtime, dict) or not isinstance(safety, dict):
        raise ValueError("Trade config runtime and safety sections must be objects.")
    if args.retry_attempts is not None:
        runtime["retry_attempts_per_state"] = args.retry_attempts
    if args.start_delay is not None:
        runtime["start_delay_seconds"] = args.start_delay

    fsm = build_trade_fsm(
        config=config,
        trade_config=trade_config,
        trade_config_path=trade_config_path,
        logger=context.logger,
    )
    fsm.source_max_buy_price = safety["source_max_buy_price"]
    fsm.min_profit_per_unit = int(safety["min_profit_per_unit"])
    fsm.start_delay_seconds = float(runtime["start_delay_seconds"])
    fsm.cycle_limit = args.cycle_limit
    if args.start_stage is not None:
        fsm.initial_state = args.start_stage
        if args.start_stage is TradeState.SELL_AT_DESTINATION:
            fsm.allow_one_unverified_sell = True
    fsm.retry_attempts_per_state = max(1, fsm.retry_attempts_per_state)
    fsm._persist_trade_config()

    start_delay = float(runtime["start_delay_seconds"])
    if start_delay > 0:
        print(f"Warning: focusing game window. Starting trade in {start_delay:.1f} seconds...")
        time.sleep(start_delay)

    try:
        result = fsm.run(context)
    except Exception as exc:
        context.logger.exception("Trade FSM failed", extra={"error": str(exc)})
        print(f"Error: {exc}")
        return 1

    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
