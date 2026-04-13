from __future__ import annotations

import argparse
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

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

DEFAULT_SOURCE_CONFIG = {
    "station_name": "Khun Port",
    "commodity_name": "Steel",
    "market_name": None,
    "is_carrier": False,
    "auto_launch_wait_seconds": 60.0,
}
DEFAULT_DESTINATION_CONFIG = {
    "station_name": "[BKRN] Event Horizon X5W-54J",
    "market_name": None,
    "is_carrier": True,
    "item_is_top": True,
    "auto_launch_wait_seconds": 45.0,
}
DEFAULT_SAFETY_CONFIG = {
    "source_max_buy_price": 7000,
    "min_profit_per_unit": 5000,
    "destination_min_demand": 1000,
}
DEFAULT_RUNTIME_CONFIG = {
    "start_delay_seconds": 3.0,
    "retry_attempts_per_state": 5,
    "retry_initial_wait_seconds": 10.0,
    "retry_max_wait_seconds": 300.0,
}


def _default_market_name(station_name: str, is_carrier: bool) -> str:
    if not is_carrier:
        return station_name

    tokens = station_name.strip().split()
    if not tokens:
        return station_name
    return tokens[-1]


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
}


@dataclass(slots=True)
class TradeFsm:
    source_station_name: str
    commodity_name: str
    destination_name: str
    market_data_source: MarketDataSource
    source_is_carrier: bool = DEFAULT_SOURCE_CONFIG["is_carrier"]
    destination_is_carrier: bool = DEFAULT_DESTINATION_CONFIG["is_carrier"]
    source_market_name: str | None = None
    destination_market_name: str | None = None
    source_max_buy_price: int | None = DEFAULT_SAFETY_CONFIG["source_max_buy_price"]
    destination_item_is_top: bool = DEFAULT_DESTINATION_CONFIG["item_is_top"]
    min_profit_per_unit: int = DEFAULT_SAFETY_CONFIG["min_profit_per_unit"]
    destination_min_demand: int = DEFAULT_SAFETY_CONFIG["destination_min_demand"]
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
    retry_attempts_per_state: int = DEFAULT_RUNTIME_CONFIG["retry_attempts_per_state"]
    retry_initial_wait_seconds: float = DEFAULT_RUNTIME_CONFIG["retry_initial_wait_seconds"]
    retry_max_wait_seconds: float = DEFAULT_RUNTIME_CONFIG["retry_max_wait_seconds"]
    source_auto_launch_wait_seconds: float = DEFAULT_SOURCE_CONFIG["auto_launch_wait_seconds"]
    destination_auto_launch_wait_seconds: float = DEFAULT_DESTINATION_CONFIG["auto_launch_wait_seconds"]
    initial_state: TradeState = TradeState.BUY_FROM_SOURCE
    last_source_buy_price: int | None = field(default=None, init=False)
    allow_one_unverified_sell: bool = field(default=False, init=False)

    name = "trade"

    def __post_init__(self) -> None:
        if self.source_market_name is None:
            self.source_market_name = _default_market_name(
                self.source_station_name,
                self.source_is_carrier,
            )
        if self.destination_market_name is None:
            self.destination_market_name = _default_market_name(
                self.destination_name,
                self.destination_is_carrier,
            )
        self.allow_one_unverified_sell = self.initial_state is TradeState.SELL_AT_DESTINATION

    def source_config(self) -> dict[str, object]:
        return {
            "station_name": self.source_station_name,
            "commodity_name": self.commodity_name,
            "market_name": self.source_market_name,
            "is_carrier": self.source_is_carrier,
            "auto_launch_wait_seconds": self.source_auto_launch_wait_seconds,
        }

    def destination_config(self) -> dict[str, object]:
        return {
            "station_name": self.destination_name,
            "market_name": self.destination_market_name,
            "is_carrier": self.destination_is_carrier,
            "item_is_top": self.destination_item_is_top,
            "auto_launch_wait_seconds": self.destination_auto_launch_wait_seconds,
        }

    def safety_config(self) -> dict[str, object]:
        return {
            "source_max_buy_price": self.source_max_buy_price,
            "min_profit_per_unit": self.min_profit_per_unit,
            "destination_min_demand": self.destination_min_demand,
            "last_source_buy_price": self.last_source_buy_price,
            "allow_one_unverified_sell": self.allow_one_unverified_sell,
        }

    def runtime_config(self) -> dict[str, object]:
        return {
            "retry_attempts_per_state": self.retry_attempts_per_state,
            "retry_initial_wait_seconds": self.retry_initial_wait_seconds,
            "retry_max_wait_seconds": self.retry_max_wait_seconds,
        }

    def run(self, context: Context) -> Result:
        state = self.initial_state
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
        if state is not TradeState.BUY_FROM_SOURCE or result.debug is None:
            return

        unit_buy_price = result.debug.get("unit_buy_price")
        if isinstance(unit_buy_price, int):
            self.last_source_buy_price = unit_buy_price

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

        if not self.destination_is_carrier and matching_item.demand < self.destination_min_demand:
            return Result.fail(
                "Destination demand is below the configured minimum; halting before sell.",
                debug={
                    "commodity": self.commodity_name,
                    "destination_name": self.destination_name,
                    "station_name": snapshot.station_name,
                    "destination_demand": matching_item.demand,
                    "destination_min_demand": self.destination_min_demand,
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
            timings=self.sell_timings,
        ).run(context)

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
            leave_timings = self.leave_station_timings
            if self.source_is_carrier and leave_timings.auto_launch_wait_seconds == LeaveStationTimings().auto_launch_wait_seconds:
                leave_timings = LeaveStationTimings(
                    auto_launch_wait_seconds=self.source_auto_launch_wait_seconds,
                    mass_lock_poll_interval_seconds=leave_timings.mass_lock_poll_interval_seconds,
                    post_mass_lock_clear_wait_seconds=leave_timings.post_mass_lock_clear_wait_seconds,
                    mass_lock_timeout_seconds=leave_timings.mass_lock_timeout_seconds,
                )
            return LeaveStation(timings=leave_timings)
        if state is TradeState.LEAVE_DESTINATION:
            leave_timings = self.leave_station_timings
            if self.destination_is_carrier and leave_timings.auto_launch_wait_seconds == LeaveStationTimings().auto_launch_wait_seconds:
                leave_timings = LeaveStationTimings(
                    auto_launch_wait_seconds=self.destination_auto_launch_wait_seconds,
                    mass_lock_poll_interval_seconds=leave_timings.mass_lock_poll_interval_seconds,
                    post_mass_lock_clear_wait_seconds=leave_timings.post_mass_lock_clear_wait_seconds,
                    mass_lock_timeout_seconds=leave_timings.mass_lock_timeout_seconds,
                )
            return LeaveStation(timings=leave_timings)
        if state is TradeState.LOCK_NAV_TO_DESTINATION:
            return LockNavDestination(
                target_name=self.destination_name,
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
                timings=self.navigation_timings,
            )
        if state is TradeState.ALIGN_TO_SOURCE:
            return AlignToTargetCompass(config=self.align_config)
        if state is TradeState.ENGAGE_FSD_TO_SOURCE:
            return EngageFsdSequence(timings=self.fsd_timings)
        if state is TradeState.DOCK_AT_SOURCE:
            return RequestDockingSequence(timings=self.docking_timings)
        raise ValueError(f"Unsupported trade state: {state!r}")

    def _next_state(self, state: TradeState, completed_cycles: int) -> tuple[TradeState, int]:
        transitions = {
            TradeState.BUY_FROM_SOURCE: TradeState.LEAVE_SOURCE,
            TradeState.LEAVE_SOURCE: TradeState.LOCK_NAV_TO_DESTINATION,
            TradeState.LOCK_NAV_TO_DESTINATION: TradeState.ALIGN_TO_DESTINATION,
            TradeState.ALIGN_TO_DESTINATION: TradeState.ENGAGE_FSD_TO_DESTINATION,
            TradeState.ENGAGE_FSD_TO_DESTINATION: TradeState.DOCK_AT_DESTINATION,
            TradeState.DOCK_AT_DESTINATION: TradeState.SELL_AT_DESTINATION,
            TradeState.SELL_AT_DESTINATION: TradeState.LEAVE_DESTINATION,
            TradeState.LEAVE_DESTINATION: TradeState.LOCK_NAV_TO_SOURCE,
            TradeState.LOCK_NAV_TO_SOURCE: TradeState.ALIGN_TO_SOURCE,
            TradeState.ALIGN_TO_SOURCE: TradeState.ENGAGE_FSD_TO_SOURCE,
            TradeState.ENGAGE_FSD_TO_SOURCE: TradeState.DOCK_AT_SOURCE,
        }
        if state is TradeState.DOCK_AT_SOURCE:
            completed_cycles += 1
            if self.cycle_limit is not None and completed_cycles >= self.cycle_limit:
                return TradeState.COMPLETED, completed_cycles
            return TradeState.BUY_FROM_SOURCE, completed_cycles
        return transitions[state], completed_cycles


@dataclass(slots=True)
class _CallableAction:
    name: str
    runner: Callable[[Context], Result]

    def run(self, context: Context) -> Result:
        return self.runner(context)


def _normalize_commodity_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def build_default_fsm(
    config: AppConfig,
    source_config: dict[str, object] | None = None,
    destination_config: dict[str, object] | None = None,
    safety_config: dict[str, object] | None = None,
    runtime_config: dict[str, object] | None = None,
    logger: logging.Logger | None = None,
) -> TradeFsm:
    source = {**DEFAULT_SOURCE_CONFIG, **(source_config or {})}
    destination = {**DEFAULT_DESTINATION_CONFIG, **(destination_config or {})}
    safety = {**DEFAULT_SAFETY_CONFIG, **(safety_config or {})}
    runtime = {**DEFAULT_RUNTIME_CONFIG, **(runtime_config or {})}
    market_reader = MarketReader(
        market_path=config.paths.market_file,
        journal_dir=config.paths.journal_dir,
        logger=logger or configure_logging(config, logger_name="elite_auto.trade"),
    )
    return TradeFsm(
        source_station_name=str(source["station_name"]),
        commodity_name=str(source["commodity_name"]),
        destination_name=str(destination["station_name"]),
        market_data_source=market_reader,
        source_is_carrier=bool(source["is_carrier"]),
        destination_is_carrier=bool(destination["is_carrier"]),
        source_market_name=source.get("market_name"),
        destination_market_name=destination.get("market_name"),
        source_max_buy_price=safety["source_max_buy_price"],
        destination_item_is_top=bool(destination["item_is_top"]),
        min_profit_per_unit=int(safety["min_profit_per_unit"]),
        destination_min_demand=int(safety["destination_min_demand"]),
        retry_attempts_per_state=int(runtime["retry_attempts_per_state"]),
        retry_initial_wait_seconds=float(runtime["retry_initial_wait_seconds"]),
        retry_max_wait_seconds=float(runtime["retry_max_wait_seconds"]),
        source_auto_launch_wait_seconds=float(source["auto_launch_wait_seconds"]),
        destination_auto_launch_wait_seconds=float(destination["auto_launch_wait_seconds"]),
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
    parser.add_argument("--source-station", default=DEFAULT_SOURCE_CONFIG["station_name"], help="Station to buy from.")
    parser.add_argument("--commodity", default=DEFAULT_SOURCE_CONFIG["commodity_name"], help="Commodity to trade.")
    parser.add_argument("--destination", default=DEFAULT_DESTINATION_CONFIG["station_name"], help="Station to sell at.")
    parser.add_argument(
        "--source-market-name",
        help="Optional station name as it appears in Market.json for the source buy location. Defaults to the carrier callsign for carrier sources.",
    )
    parser.add_argument(
        "--source-max-buy-price",
        type=int,
        default=DEFAULT_SAFETY_CONFIG["source_max_buy_price"],
        help="Proceed with source buy despite a station-name mismatch if the requested commodity is buyable at or below this price.",
    )
    parser.add_argument(
        "--source-is-carrier",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SOURCE_CONFIG["is_carrier"],
        help="Whether the source buy location uses the carrier services layout. Destination sell stays on station flow.",
    )
    parser.add_argument(
        "--source-auto-launch-wait-seconds",
        type=float,
        default=DEFAULT_SOURCE_CONFIG["auto_launch_wait_seconds"],
        help="Seconds to wait after auto-launch when leaving the source location if it is a carrier.",
    )
    parser.add_argument(
        "--destination-item-is-top",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DESTINATION_CONFIG["item_is_top"],
        help="Whether the destination sell commodity is always the top row, allowing filter steps to be skipped.",
    )
    parser.add_argument(
        "--destination-is-carrier",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DESTINATION_CONFIG["is_carrier"],
        help="Whether the destination sell location uses the carrier services layout.",
    )
    parser.add_argument(
        "--destination-auto-launch-wait-seconds",
        type=float,
        default=DEFAULT_DESTINATION_CONFIG["auto_launch_wait_seconds"],
        help="Seconds to wait after auto-launch when leaving the destination location if it is a carrier.",
    )
    parser.add_argument(
        "--destination-market-name",
        help="Optional station name as it appears in Market.json for the destination sell location. Defaults to the carrier callsign for carrier destinations.",
    )
    parser.add_argument(
        "--min-profit-per-unit",
        type=int,
        default=DEFAULT_SAFETY_CONFIG["min_profit_per_unit"],
        help="Minimum credits of profit required per unit before destination sell is allowed.",
    )
    parser.add_argument(
        "--destination-min-demand",
        type=int,
        default=DEFAULT_SAFETY_CONFIG["destination_min_demand"],
        help="Minimum destination demand required before selling at a non-carrier station.",
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
        default=DEFAULT_RUNTIME_CONFIG["retry_attempts_per_state"],
        help="Attempts per FSM state before quitting. Default: 5.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=DEFAULT_RUNTIME_CONFIG["start_delay_seconds"],
        help="Seconds to wait before starting so the game window can be focused.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.from_json(args.config) if args.config else AppConfig.default()
    context = build_context(config)
    source_config = {
        "station_name": args.source_station,
        "commodity_name": args.commodity,
        "market_name": args.source_market_name,
        "is_carrier": args.source_is_carrier,
        "auto_launch_wait_seconds": args.source_auto_launch_wait_seconds,
    }
    destination_config = {
        "station_name": args.destination,
        "market_name": args.destination_market_name,
        "is_carrier": args.destination_is_carrier,
        "item_is_top": args.destination_item_is_top,
        "auto_launch_wait_seconds": args.destination_auto_launch_wait_seconds,
    }
    safety_config = {
        "source_max_buy_price": args.source_max_buy_price,
        "min_profit_per_unit": args.min_profit_per_unit,
        "destination_min_demand": args.destination_min_demand,
    }
    runtime_config = {
        "retry_attempts_per_state": args.retry_attempts,
        "retry_initial_wait_seconds": DEFAULT_RUNTIME_CONFIG["retry_initial_wait_seconds"],
        "retry_max_wait_seconds": DEFAULT_RUNTIME_CONFIG["retry_max_wait_seconds"],
    }
    fsm = build_default_fsm(
        config=config,
        source_config=source_config,
        destination_config=destination_config,
        safety_config=safety_config,
        runtime_config=runtime_config,
        logger=context.logger,
    )
    fsm.cycle_limit = args.cycle_limit
    if args.start_stage is not None:
        fsm.initial_state = args.start_stage
        if args.start_stage is TradeState.SELL_AT_DESTINATION:
            fsm.allow_one_unverified_sell = True
    fsm.retry_attempts_per_state = max(1, fsm.retry_attempts_per_state)

    if args.start_delay > 0:
        print(f"Warning: focusing game window. Starting trade in {args.start_delay:.1f} seconds...")
        time.sleep(args.start_delay)

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
