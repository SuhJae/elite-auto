from __future__ import annotations

import argparse
import logging
import math
import time
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
from app.domain.protocols import MarketDataSource
from app.domain.result import Result
from app.state.market_reader import MarketReader
from run import build_context

DEFAULT_SOURCE_STATION_NAME = "P.T.N. BLUE TRADER BZL-59X"
DEFAULT_COMMODITY_NAME = "AGRONOMIC TREATMENT"
DEFAULT_DESTINATION_NAME = "MILLE GATEWAY"
DEFAULT_START_DELAY_SECONDS = 3.0
DEFAULT_RETRY_ATTEMPTS_PER_STATE = 5
DEFAULT_RETRY_INITIAL_WAIT_SECONDS = 10.0
DEFAULT_RETRY_MAX_WAIT_SECONDS = 300.0
DEFAULT_SOURCE_IS_CARRIER = True
DEFAULT_DESTINATION_ITEM_IS_TOP = True
DEFAULT_SOURCE_MAX_BUY_PRICE = 15500


def _default_source_market_name(source_station_name: str, source_is_carrier: bool) -> str:
    if not source_is_carrier:
        return source_station_name

    tokens = source_station_name.strip().split()
    if not tokens:
        return source_station_name
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
    source_is_carrier: bool = DEFAULT_SOURCE_IS_CARRIER
    source_market_name: str | None = None
    source_max_buy_price: int | None = DEFAULT_SOURCE_MAX_BUY_PRICE
    destination_item_is_top: bool = DEFAULT_DESTINATION_ITEM_IS_TOP
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
    retry_attempts_per_state: int = DEFAULT_RETRY_ATTEMPTS_PER_STATE
    retry_initial_wait_seconds: float = DEFAULT_RETRY_INITIAL_WAIT_SECONDS
    retry_max_wait_seconds: float = DEFAULT_RETRY_MAX_WAIT_SECONDS
    initial_state: TradeState = TradeState.BUY_FROM_SOURCE

    name = "trade"

    def __post_init__(self) -> None:
        if self.source_market_name is None:
            self.source_market_name = _default_source_market_name(
                self.source_station_name,
                self.source_is_carrier,
            )

    def run(self, context: Context) -> Result:
        state = self.initial_state
        history: list[dict[str, object]] = []
        completed_cycles = 0
        context.logger.info(
            "Starting trade FSM",
            extra={
                "source_station_name": self.source_station_name,
                "commodity_name": self.commodity_name,
                "destination_name": self.destination_name,
                "source_is_carrier": self.source_is_carrier,
                "source_market_name": self.source_market_name,
                "source_max_buy_price": self.source_max_buy_price,
                "destination_item_is_top": self.destination_item_is_top,
                "initial_state": state.value,
                "cycle_limit": self.cycle_limit,
                "retry_attempts_per_state": self.retry_attempts_per_state,
                "retry_initial_wait_seconds": self.retry_initial_wait_seconds,
                "retry_max_wait_seconds": self.retry_max_wait_seconds,
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
                        "source_station_name": self.source_station_name,
                        "commodity_name": self.commodity_name,
                        "destination_name": self.destination_name,
                        "source_is_carrier": self.source_is_carrier,
                        "source_market_name": self.source_market_name,
                        "source_max_buy_price": self.source_max_buy_price,
                        "destination_item_is_top": self.destination_item_is_top,
                        "completed_cycles": completed_cycles,
                    },
                )

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
                "source_station_name": self.source_station_name,
                "commodity_name": self.commodity_name,
                "destination_name": self.destination_name,
                "source_is_carrier": self.source_is_carrier,
                "source_market_name": self.source_market_name,
                "source_max_buy_price": self.source_max_buy_price,
                "destination_item_is_top": self.destination_item_is_top,
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
            return LeaveStation(timings=self.leave_station_timings)
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
            return SellFromStarport(
                station_name=self.destination_name,
                commodity=self.commodity_name,
                market_data_source=self.market_data_source,
                is_top=self.destination_item_is_top,
                timings=self.sell_timings,
            )
        if state is TradeState.LEAVE_DESTINATION:
            return LeaveStation(timings=self.leave_station_timings)
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


def build_default_fsm(
    config: AppConfig,
    source_station_name: str = DEFAULT_SOURCE_STATION_NAME,
    commodity_name: str = DEFAULT_COMMODITY_NAME,
    destination_name: str = DEFAULT_DESTINATION_NAME,
    source_is_carrier: bool = DEFAULT_SOURCE_IS_CARRIER,
    source_market_name: str | None = None,
    source_max_buy_price: int | None = DEFAULT_SOURCE_MAX_BUY_PRICE,
    destination_item_is_top: bool = DEFAULT_DESTINATION_ITEM_IS_TOP,
    logger: logging.Logger | None = None,
) -> TradeFsm:
    market_reader = MarketReader(
        market_path=config.paths.market_file,
        journal_dir=config.paths.journal_dir,
        logger=logger or configure_logging(config, logger_name="elite_auto.trade"),
    )
    return TradeFsm(
        source_station_name=source_station_name,
        commodity_name=commodity_name,
        destination_name=destination_name,
        market_data_source=market_reader,
        source_is_carrier=source_is_carrier,
        source_market_name=source_market_name,
        source_max_buy_price=source_max_buy_price,
        destination_item_is_top=destination_item_is_top,
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
    parser.add_argument("--source-station", default=DEFAULT_SOURCE_STATION_NAME, help="Station to buy from.")
    parser.add_argument("--commodity", default=DEFAULT_COMMODITY_NAME, help="Commodity to trade.")
    parser.add_argument("--destination", default=DEFAULT_DESTINATION_NAME, help="Station to sell at.")
    parser.add_argument(
        "--source-market-name",
        help="Optional station name as it appears in Market.json for the source buy location. Defaults to the carrier callsign for carrier sources.",
    )
    parser.add_argument(
        "--source-max-buy-price",
        type=int,
        default=DEFAULT_SOURCE_MAX_BUY_PRICE,
        help="Proceed with source buy despite a station-name mismatch if the requested commodity is buyable at or below this price.",
    )
    parser.add_argument(
        "--source-is-carrier",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SOURCE_IS_CARRIER,
        help="Whether the source buy location uses the carrier services layout. Destination sell stays on station flow.",
    )
    parser.add_argument(
        "--destination-item-is-top",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DESTINATION_ITEM_IS_TOP,
        help="Whether the destination sell commodity is always the top row, allowing filter steps to be skipped.",
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
        default=DEFAULT_RETRY_ATTEMPTS_PER_STATE,
        help="Attempts per FSM state before quitting. Default: 5.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=DEFAULT_START_DELAY_SECONDS,
        help="Seconds to wait before starting so the game window can be focused.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.from_json(args.config) if args.config else AppConfig.default()
    context = build_context(config)
    fsm = build_default_fsm(
        config=config,
        source_station_name=args.source_station,
        commodity_name=args.commodity,
        destination_name=args.destination,
        source_is_carrier=args.source_is_carrier,
        source_market_name=args.source_market_name,
        source_max_buy_price=args.source_max_buy_price,
        destination_item_is_top=args.destination_item_is_top,
        logger=context.logger,
    )
    fsm.cycle_limit = args.cycle_limit
    if args.start_stage is not None:
        fsm.initial_state = args.start_stage
    fsm.retry_attempts_per_state = max(1, args.retry_attempts)

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
