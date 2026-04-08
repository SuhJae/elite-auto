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
from app.actions.leave_construction import LeaveConstruction, LeaveConstructionTimings
from app.actions.leave_station import LeaveStation, LeaveStationTimings
from app.actions.navigation import LockNavDestination, NavigationTimings
from app.actions.starport_buy import BuyFromStarport, StarportBuyTimings
from app.actions.unload_construction import UnloadConstruction, UnloadConstructionTimings

from app.config import AppConfig, configure_logging
from app.domain.context import Context
from app.domain.protocols import MarketDataSource
from app.domain.result import Result
from app.state.market_reader import MarketReader
from run import build_context

DEFAULT_STATION_NAME = "HILDEBRANDT REFINERY"
# DEFAULT_STATION_NAME = "VERHOEVEN REFINERY"
# DEFAULT_COMMODITY_NAME = "LIQUID OXYGEN"
DEFAULT_COMMODITY_NAME = "STEEL"
DEFAULT_TARGET_NAME = "ORBITAL CONSTRUCTION SITE: PARISE GATEWAY"
# DEFAULT_TARGET_NAME = "ORBITAL CONSTRUCTION SITE: PRATCHETT POINT"
DEFAULT_START_DELAY_SECONDS = 3.0
DEFAULT_RETRY_ATTEMPTS_PER_STATE = 5
DEFAULT_RETRY_INITIAL_WAIT_SECONDS = 10.0
DEFAULT_RETRY_MAX_WAIT_SECONDS = 300.0


class ColonizeState(str, Enum):
    BUY_LIQUID_OXYGEN = "buy_liquid_oxygen"
    LEAVE_STATION = "leave_station"
    LOCK_NAV_TO_CONSTRUCTION = "lock_nav_to_construction"
    ALIGN_TO_CONSTRUCTION = "align_to_construction"
    ENGAGE_FSD_TO_CONSTRUCTION = "engage_fsd_to_construction"
    DOCK_AT_CONSTRUCTION = "dock_at_construction"
    UNLOAD_CONSTRUCTION = "unload_construction"
    LEAVE_CONSTRUCTION = "leave_construction"
    LOCK_NAV_TO_REFINERY = "lock_nav_to_refinery"
    ALIGN_TO_REFINERY = "align_to_refinery"
    ENGAGE_FSD_TO_REFINERY = "engage_fsd_to_refinery"
    ALIGN_BEFORE_REFINERY_DOCK = "align_before_refinery_dock"
    DOCK_AT_REFINERY = "dock_at_refinery"
    COMPLETED = "completed"


STAGE_ALIASES: dict[str, ColonizeState] = {
    "buy": ColonizeState.BUY_LIQUID_OXYGEN,
    "buy-station": ColonizeState.BUY_LIQUID_OXYGEN,
    "leave-station": ColonizeState.LEAVE_STATION,
    "nav-construction": ColonizeState.LOCK_NAV_TO_CONSTRUCTION,
    "lock-nav-construction": ColonizeState.LOCK_NAV_TO_CONSTRUCTION,
    "align-construction": ColonizeState.ALIGN_TO_CONSTRUCTION,
    "fsd-construction": ColonizeState.ENGAGE_FSD_TO_CONSTRUCTION,
    "dock-construction": ColonizeState.DOCK_AT_CONSTRUCTION,
    "unload-construction": ColonizeState.UNLOAD_CONSTRUCTION,
    "leave-construction": ColonizeState.LEAVE_CONSTRUCTION,
    "nav-station": ColonizeState.LOCK_NAV_TO_REFINERY,
    "lock-nav-station": ColonizeState.LOCK_NAV_TO_REFINERY,
    "align-station": ColonizeState.ALIGN_TO_REFINERY,
    "fsd-station": ColonizeState.ENGAGE_FSD_TO_REFINERY,
    "align-before-dock-station": ColonizeState.ALIGN_BEFORE_REFINERY_DOCK,
    "dock-station": ColonizeState.DOCK_AT_REFINERY,
    "dock-refinery": ColonizeState.DOCK_AT_REFINERY,
}


@dataclass(slots=True)
class ColonizeFsm:
    station_name: str
    commodity_name: str
    target_name: str
    market_data_source: MarketDataSource
    buy_timings: StarportBuyTimings = field(default_factory=StarportBuyTimings)
    leave_station_timings: LeaveStationTimings = field(default_factory=LeaveStationTimings)
    leave_construction_timings: LeaveConstructionTimings = field(default_factory=LeaveConstructionTimings)
    navigation_timings: NavigationTimings = field(default_factory=NavigationTimings)
    align_config: AlignConfig = field(default_factory=AlignConfig)
    pre_dock_align_config: AlignConfig = field(
        default_factory=lambda: AlignConfig(
            alignment_tolerance_px=4.0,
            axis_alignment_tolerance_px=4.0,
            confirmation_reads=1,
        )
    )
    fsd_timings: FsdTimings = field(default_factory=FsdTimings)
    docking_timings: DockingTimings = field(default_factory=DockingTimings)
    unload_construction_timings: UnloadConstructionTimings = field(default_factory=UnloadConstructionTimings)
    cycle_limit: int | None = None
    retry_attempts_per_state: int = DEFAULT_RETRY_ATTEMPTS_PER_STATE
    retry_initial_wait_seconds: float = DEFAULT_RETRY_INITIAL_WAIT_SECONDS
    retry_max_wait_seconds: float = DEFAULT_RETRY_MAX_WAIT_SECONDS
    initial_state: ColonizeState = ColonizeState.BUY_LIQUID_OXYGEN

    name = "colonize"

    def run(self, context: Context) -> Result:
        state = self.initial_state
        history: list[dict[str, object]] = []
        completed_cycles = 0
        context.logger.info(
            "Starting colonize FSM",
            extra={
                "station_name": self.station_name,
                "commodity_name": self.commodity_name,
                "construction_target_name": self.target_name,
                "initial_state": state.value,
                "cycle_limit": self.cycle_limit,
                "retry_attempts_per_state": self.retry_attempts_per_state,
                "retry_initial_wait_seconds": self.retry_initial_wait_seconds,
                "retry_max_wait_seconds": self.retry_max_wait_seconds,
            },
        )

        while state is not ColonizeState.COMPLETED:
            action = self._build_action(state)
            result, state_history = self._run_state_with_retries(context, state, action)
            history.extend(state_history)

            if not result.success:
                return Result.fail(
                    f"Colonize FSM failed in state '{state.value}': {result.reason}",
                    debug={
                        "failed_state": state.value,
                        "history": history,
                        "step_debug": result.debug,
                        "last_attempt": state_history[-1] if state_history else None,
                        "station_name": self.station_name,
                        "commodity_name": self.commodity_name,
                        "construction_target_name": self.target_name,
                        "completed_cycles": completed_cycles,
                    },
                )

            next_state, completed_cycles = self._next_state(state, completed_cycles)
            context.logger.info(
                "Colonize FSM transition",
                extra={
                    "from_state": state.value,
                    "to_state": next_state.value,
                    "completed_cycles": completed_cycles,
                },
            )
            state = next_state

        return Result.ok(
            "Colonize FSM completed.",
            debug={
                "history": history,
                "station_name": self.station_name,
                "commodity_name": self.commodity_name,
                "construction_target_name": self.target_name,
                "completed_cycles": completed_cycles,
            },
        )

    def _run_state_with_retries(
        self,
        context: Context,
        state: ColonizeState,
        action: object,
    ) -> tuple[Result, list[dict[str, object]]]:
        total_attempts = max(1, self.retry_attempts_per_state)
        state_history: list[dict[str, object]] = []

        for attempt in range(1, total_attempts + 1):
            context.logger.info(
                "Colonize FSM state start",
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
                    "Colonize FSM state raised exception",
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
                "Colonize FSM state failed; retrying",
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

    def _build_action(self, state: ColonizeState):
        if state is ColonizeState.BUY_LIQUID_OXYGEN:
            return BuyFromStarport(
                station_name=self.station_name,
                commodity=self.commodity_name,
                market_data_source=self.market_data_source,
                timings=self.buy_timings,
            )
        if state is ColonizeState.LEAVE_STATION:
            return LeaveStation(timings=self.leave_station_timings)
        if state is ColonizeState.LOCK_NAV_TO_CONSTRUCTION:
            return LockNavDestination(
                target_name=self.target_name,
                timings=self.navigation_timings,
            )
        if state is ColonizeState.ALIGN_TO_CONSTRUCTION:
            return AlignToTargetCompass(config=self.align_config)
        if state is ColonizeState.ENGAGE_FSD_TO_CONSTRUCTION:
            return EngageFsdSequence(timings=self.fsd_timings)
        if state is ColonizeState.DOCK_AT_CONSTRUCTION:
            return RequestDockingSequence(timings=self.docking_timings)
        if state is ColonizeState.UNLOAD_CONSTRUCTION:
            return UnloadConstruction(timings=self.unload_construction_timings)
        if state is ColonizeState.LEAVE_CONSTRUCTION:
            return LeaveConstruction(timings=self.leave_construction_timings)
        if state is ColonizeState.LOCK_NAV_TO_REFINERY:
            return LockNavDestination(
                target_name=self.station_name,
                timings=self.navigation_timings,
            )
        if state is ColonizeState.ALIGN_TO_REFINERY:
            return AlignToTargetCompass(config=self.align_config)
        if state is ColonizeState.ENGAGE_FSD_TO_REFINERY:
            return EngageFsdSequence(timings=self.fsd_timings)
        if state is ColonizeState.ALIGN_BEFORE_REFINERY_DOCK:
            return AlignToTargetCompass(config=self.pre_dock_align_config)
        if state is ColonizeState.DOCK_AT_REFINERY:
            return RequestDockingSequence(timings=self.docking_timings)
        raise ValueError(f"Unsupported colonize state: {state!r}")

    def _next_state(self, state: ColonizeState, completed_cycles: int) -> tuple[ColonizeState, int]:
        transitions = {
            ColonizeState.BUY_LIQUID_OXYGEN: ColonizeState.LEAVE_STATION,
            ColonizeState.LEAVE_STATION: ColonizeState.LOCK_NAV_TO_CONSTRUCTION,
            ColonizeState.LOCK_NAV_TO_CONSTRUCTION: ColonizeState.ALIGN_TO_CONSTRUCTION,
            ColonizeState.ALIGN_TO_CONSTRUCTION: ColonizeState.ENGAGE_FSD_TO_CONSTRUCTION,
            ColonizeState.ENGAGE_FSD_TO_CONSTRUCTION: ColonizeState.DOCK_AT_CONSTRUCTION,
            ColonizeState.DOCK_AT_CONSTRUCTION: ColonizeState.UNLOAD_CONSTRUCTION,
            ColonizeState.UNLOAD_CONSTRUCTION: ColonizeState.LEAVE_CONSTRUCTION,
            ColonizeState.LEAVE_CONSTRUCTION: ColonizeState.LOCK_NAV_TO_REFINERY,
            ColonizeState.LOCK_NAV_TO_REFINERY: ColonizeState.ALIGN_TO_REFINERY,
            ColonizeState.ALIGN_TO_REFINERY: ColonizeState.ENGAGE_FSD_TO_REFINERY,
            ColonizeState.ENGAGE_FSD_TO_REFINERY: ColonizeState.ALIGN_BEFORE_REFINERY_DOCK,
            ColonizeState.ALIGN_BEFORE_REFINERY_DOCK: ColonizeState.DOCK_AT_REFINERY,
        }
        if state is ColonizeState.DOCK_AT_REFINERY:
            completed_cycles += 1
            if self.cycle_limit is not None and completed_cycles >= self.cycle_limit:
                return ColonizeState.COMPLETED, completed_cycles
            return ColonizeState.BUY_LIQUID_OXYGEN, completed_cycles
        return transitions[state], completed_cycles


def build_default_fsm(
    config: AppConfig,
    station_name: str = DEFAULT_STATION_NAME,
    commodity_name: str = DEFAULT_COMMODITY_NAME,
    target_name: str = DEFAULT_TARGET_NAME,
    logger: logging.Logger | None = None,
) -> ColonizeFsm:
    market_reader = MarketReader(
        market_path=config.paths.market_file,
        journal_dir=config.paths.journal_dir,
        logger=logger or configure_logging(config, logger_name="elite_auto.colonize"),
    )
    return ColonizeFsm(
        station_name=station_name,
        commodity_name=commodity_name,
        target_name=target_name,
        market_data_source=market_reader,
    )


def parse_start_stage(value: str) -> ColonizeState:
    normalized = value.strip().lower().replace("_", "-")
    alias_match = STAGE_ALIASES.get(normalized)
    if alias_match is not None:
        return alias_match

    try:
        return ColonizeState(normalized.replace("-", "_"))
    except ValueError as exc:
        available = ", ".join(sorted(STAGE_ALIASES))
        raise argparse.ArgumentTypeError(
            f"Unknown start stage '{value}'. Available aliases: {available}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the colonize FSM.")
    parser.add_argument("--config", help="Optional path to a JSON config file.")
    parser.add_argument("--station", default=DEFAULT_STATION_NAME, help="Station to buy from.")
    parser.add_argument("--commodity", default=DEFAULT_COMMODITY_NAME, help="Commodity to buy.")
    parser.add_argument("--target", default=DEFAULT_TARGET_NAME, help="Navigation target to lock.")
    parser.add_argument(
        "--start-stage",
        type=parse_start_stage,
        help="Optional stage to resume from, for example dock-station or unload-construction.",
    )
    parser.add_argument(
        "--cycle-limit",
        type=int,
        help="Optional number of full refinery-to-refinery loops to run. Omit for infinite looping.",
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
        station_name=args.station,
        commodity_name=args.commodity,
        target_name=args.target,
        logger=context.logger,
    )
    fsm.cycle_limit = args.cycle_limit
    if args.start_stage is not None:
        fsm.initial_state = args.start_stage
    fsm.retry_attempts_per_state = max(1, args.retry_attempts)

    if args.start_delay > 0:
        print(f"Warning: focusing game window. Starting colonize in {args.start_delay:.1f} seconds...")
        time.sleep(args.start_delay)

    try:
        result = fsm.run(context)
    except Exception as exc:
        context.logger.exception("Colonize FSM failed", extra={"error": str(exc)})
        print(f"Error: {exc}")
        return 1

    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
