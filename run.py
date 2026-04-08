from __future__ import annotations

import argparse
import json
import sys
from typing import Callable

from app.actions.fsd import WaitForMassLockClear
from app.actions.leave_station import LeaveStation
from app.actions.launch import WaitUntilUndocked
from app.actions.navigation import LockNavDestination, OpenNavPanel, WaitForSupercruiseEntry
from app.actions.navigation_ocr import MoveCursorToNavTarget
from app.actions.starport_buy import BuyFromStarport, print_buy_screen
from app.adapters.capture_dxcam import DxcamCapture
from app.adapters.input_pydirect import PydirectInputAdapter, PydirectInputShipControl, ShipKeyMap
from app.adapters.vision_cv import OpenCVVisionSystem
from app.config import AppConfig, configure_logging
from app.domain.context import Context
from app.domain.result import Result
from app.routines.source_cycle import SampleDepartureRoutine
from app.state.bindings_reader import EliteBindingsReader
from app.state.cargo_reader import CargoReader
from app.state.journal_tailer import JournalTailer
from app.state.market_reader import MarketReader
from app.state.status_reader import EliteStateReader, StatusFileReader


def build_context(config: AppConfig) -> Context:
    logger = configure_logging(config)
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

    if detected_bindings is not None:
        logger.info(
            "Elite bindings auto-detected",
            extra={
                "preset_name": detected_bindings.preset_name,
                "binds_path": str(detected_bindings.binds_path),
                "detected_fields": detected_bindings.detected_fields,
            },
        )

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
    input_adapter = PydirectInputAdapter()
    ship_control = PydirectInputShipControl(input_adapter, keymap)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Elite Auto architecture scaffold runner")
    parser.add_argument("--config", help="Optional path to a JSON config file.")
    parser.add_argument("--action", help="Run one action by name.")
    parser.add_argument("--routine", help="Run one routine by name.")
    parser.add_argument("--station", help="Expected station name for station-market safety checks.")
    parser.add_argument("--target", help="Navigation target name for nav actions.")
    parser.add_argument(
        "--commodity",
        help="Commodity name for station trading actions.",
    )
    parser.add_argument("--status", action="store_true", help="Print the current state snapshot.")
    parser.add_argument("--market", action="store_true", help="Print the current parsed market snapshot.")
    parser.add_argument("--buy-screen", action="store_true", help="Print the simulated starport Buy screen.")
    return parser.parse_args()


def action_registry(config: AppConfig) -> dict[str, Callable[[], object]]:
    timeout = config.runtime.action_timeout_seconds
    poll = config.runtime.poll_interval_seconds
    return {
        OpenNavPanel.name: lambda: OpenNavPanel(),
        LockNavDestination.name: lambda: LockNavDestination(target_name=""),
        MoveCursorToNavTarget.name: lambda: MoveCursorToNavTarget(target_name=""),
        WaitUntilUndocked.name: lambda: WaitUntilUndocked(timeout_seconds=timeout, poll_interval_seconds=poll),
        LeaveStation.name: lambda: LeaveStation(),
        WaitForMassLockClear.name: lambda: WaitForMassLockClear(
            timeout_seconds=timeout,
            poll_interval_seconds=poll,
        ),
        WaitForSupercruiseEntry.name: lambda: WaitForSupercruiseEntry(
            timeout_seconds=timeout,
            poll_interval_seconds=poll,
        ),
    }


def routine_registry(config: AppConfig) -> dict[str, Callable[[], object]]:
    timeout = config.runtime.action_timeout_seconds
    poll = config.runtime.poll_interval_seconds
    return {
        SampleDepartureRoutine.name: lambda: SampleDepartureRoutine(
            timeout_seconds=timeout,
            poll_interval_seconds=poll,
        )
    }


def print_status(context: Context) -> int:
    state = context.state_reader.snapshot()
    print(json.dumps(state.to_debug_dict(), indent=2))
    return 0


def print_market(context: Context) -> int:
    market_reader = MarketReader(
        market_path=context.config.paths.market_file,
        journal_dir=context.config.paths.journal_dir,
        logger=context.logger,
    )
    snapshot = market_reader.snapshot(required=True)
    state = context.state_reader.snapshot()
    payload = snapshot.to_dict()
    payload["currently_docked"] = state.is_docked
    print(json.dumps(payload, indent=2))
    return 0


def print_market_buy_screen(context: Context) -> int:
    market_reader = MarketReader(
        market_path=context.config.paths.market_file,
        journal_dir=context.config.paths.journal_dir,
        logger=context.logger,
    )
    snapshot = market_reader.snapshot(required=True)
    print_buy_screen(snapshot)
    return 0


def run_named_action(name: str, context: Context) -> int:
    factory = action_registry(context.config).get(name)
    if factory is None:
        print(f"Unknown action: {name}", file=sys.stderr)
        return 2
    result = factory().run(context)
    return _print_result(result)


def run_named_action_with_args(
    name: str,
    commodity_name: str | None,
    station_name: str | None,
    target_name: str | None,
    context: Context,
) -> int:
    if name == BuyFromStarport.name:
        if not station_name:
            print("The --station argument is required for buy_from_starport.", file=sys.stderr)
            return 2
        if not commodity_name:
            print("The --commodity argument is required for buy_from_starport.", file=sys.stderr)
            return 2
        market_reader = MarketReader(
            market_path=context.config.paths.market_file,
            journal_dir=context.config.paths.journal_dir,
            logger=context.logger,
        )
        action = BuyFromStarport(
            station_name=station_name,
            commodity=commodity_name,
            market_data_source=market_reader,
        )
        result = action.run(context)
        return _print_result(result)
    if name == MoveCursorToNavTarget.name:
        if not target_name:
            print("The --target argument is required for move_cursor_to_nav_target.", file=sys.stderr)
            return 2
        action = MoveCursorToNavTarget(target_name=target_name)
        result = action.run(context)
        return _print_result(result)
    if name == LockNavDestination.name:
        if not target_name:
            print("The --target argument is required for lock_nav_destination.", file=sys.stderr)
            return 2
        action = LockNavDestination(target_name=target_name)
        result = action.run(context)
        return _print_result(result)
    return run_named_action(name, context)


def run_named_routine(name: str, context: Context) -> int:
    factory = routine_registry(context.config).get(name)
    if factory is None:
        print(f"Unknown routine: {name}", file=sys.stderr)
        return 2
    result = factory().run(context)
    return _print_result(result)


def _print_result(result: Result) -> int:
    print(f"Success: {result.success}")
    print(f"Reason: {result.reason}")
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


def main() -> int:
    args = parse_args()
    if not any([args.status, args.market, args.buy_screen, args.action, args.routine]):
        print("Pass one of --status, --market, --buy-screen, --action, or --routine.", file=sys.stderr)
        return 2

    config = AppConfig.from_json(args.config) if args.config else AppConfig.default()
    context = build_context(config)

    try:
        if args.status:
            return print_status(context)
        if args.market:
            return print_market(context)
        if args.buy_screen:
            return print_market_buy_screen(context)
        if args.action:
            return run_named_action_with_args(args.action, args.commodity, args.station, args.target, context)
        if args.routine:
            return run_named_routine(args.routine, context)
    except Exception as exc:
        context.logger.exception("Command failed", extra={"error": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
