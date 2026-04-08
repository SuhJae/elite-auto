from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.actions.starport_buy import build_standalone_context
from app.config import AppConfig
from app.domain.context import Context
from app.domain.result import Result


# Edit these values for standalone testing of this file.
STANDALONE_CONFIG_PATH: str | None = None
STANDALONE_ACTION = "engage_fsd_sequence"
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_POST_MENU_OPEN_WAIT_SECONDS = 2.0
STANDALONE_POST_SELECT_WAIT_SECONDS = 1.0
STANDALONE_KEY_INTERVAL_SECONDS = 0.5
STANDALONE_POST_BACK_WAIT_SECONDS = 0.5
STANDALONE_SUPERCRUISE_ENTRY_TIMEOUT_SECONDS = 60.0
STANDALONE_POLL_INTERVAL_SECONDS = 0.5
STANDALONE_SAFETY_TIMEOUT_SECONDS = 15 * 60.0
STANDALONE_CANCEL_WAIT_SECONDS = 1.0


@dataclass(slots=True)
class FsdTimings:
    """Named timings for the aligned-then-engage FSD sequence."""

    post_menu_open_wait_seconds: float = 2.0
    post_select_wait_seconds: float = 1.0
    key_interval_seconds: float = 0.5
    post_back_wait_seconds: float = 0.5
    supercruise_entry_timeout_seconds: float = 60.0
    poll_interval_seconds: float = 0.5
    safety_timeout_seconds: float = 15 * 60.0
    cancel_wait_seconds: float = 1.0


@dataclass(slots=True)
class EngageFsdSequence:
    """Engage FSD, perform the requested nav-menu sequence, then wait for disengage."""

    timings: FsdTimings = field(default_factory=FsdTimings)

    name = "engage_fsd_sequence"

    def run(self, context: Context) -> Result:
        ship_control = context.ship_control
        input_adapter = context.input_adapter
        if ship_control is None or input_adapter is None:
            return Result.fail("FSD sequence requires both ship control and input control.")

        initial_state = context.state_reader.snapshot()
        if initial_state.is_docked:
            return Result.fail(
                "Cannot run the FSD sequence while docked.",
                debug={"state": initial_state.to_debug_dict()},
            )

        context.logger.info(
            "Starting FSD sequence",
            extra={
                "post_menu_open_wait_seconds": self.timings.post_menu_open_wait_seconds,
                "post_select_wait_seconds": self.timings.post_select_wait_seconds,
                "key_interval_seconds": self.timings.key_interval_seconds,
                "safety_timeout_seconds": self.timings.safety_timeout_seconds,
            },
        )

        ship_control.charge_fsd("supercruise")
        ship_control.set_throttle_percent(100)

        entry_result = WaitForSupercruiseEntry(
            timeout_seconds=self.timings.supercruise_entry_timeout_seconds,
            poll_interval_seconds=self.timings.poll_interval_seconds,
        ).run(context)
        if not entry_result.success:
            return Result.fail(
                "FSD sequence failed before supercruise entry.",
                debug=entry_result.debug,
            )

        self._run_nav_menu_sequence(context)

        disengage_result = WaitForFsdDisengage(
            timeout_seconds=self.timings.safety_timeout_seconds,
            poll_interval_seconds=self.timings.poll_interval_seconds,
        ).run(context)
        if disengage_result.success:
            return Result.ok(
                "FSD sequence completed and the ship left supercruise.",
                debug={
                    "initial_state": initial_state.to_debug_dict(),
                    "supercruise_entry": entry_result.debug,
                    "fsd_disengage": disengage_result.debug,
                },
            )

        context.logger.warning(
            "FSD safety timeout reached; cancelling FSD",
            extra={"timeout_seconds": self.timings.safety_timeout_seconds},
        )
        ship_control.charge_fsd("supercruise")
        time.sleep(self.timings.cancel_wait_seconds)

        final_state = context.state_reader.snapshot()
        return Result.fail(
            "FSD safety timeout reached; sent FSD cancel input.",
            debug={
                "initial_state": initial_state.to_debug_dict(),
                "supercruise_entry": entry_result.debug,
                "fsd_disengage": disengage_result.debug,
                "final_state": final_state.to_debug_dict(),
                "safety_timeout_seconds": self.timings.safety_timeout_seconds,
            },
        )

    def _run_nav_menu_sequence(self, context: Context) -> None:
        ship_control = context.ship_control
        input_adapter = context.input_adapter
        if ship_control is None or input_adapter is None:
            raise RuntimeError("Input control is not available.")

        ship_control.open_left_panel()
        time.sleep(self.timings.post_menu_open_wait_seconds)
        ship_control.ui_select("select")
        time.sleep(self.timings.post_select_wait_seconds)
        ship_control.ui_select("right")
        time.sleep(self.timings.key_interval_seconds)
        ship_control.ui_select("select")
        time.sleep(self.timings.key_interval_seconds)
        input_adapter.press(context.config.controls.ui_back)
        time.sleep(self.timings.post_back_wait_seconds)


class WaitForMassLockClear:
    """Poll state until mass lock clears."""

    name = "wait_for_mass_lock_clear"

    def __init__(self, timeout_seconds: float, poll_interval_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def run(self, context: Context) -> Result:
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            state = context.state_reader.snapshot()
            if not state.is_mass_locked:
                context.logger.info("Mass lock cleared", extra={"state": state.to_debug_dict()})
                return Result.ok("Mass lock cleared.", debug=state.to_debug_dict())
            time.sleep(self._poll_interval_seconds)

        return Result.fail(
            "Timed out waiting for mass lock to clear.",
            debug={"timeout_seconds": self._timeout_seconds},
        )


class WaitForSupercruiseEntry:
    """Poll state until supercruise becomes active."""

    name = "wait_for_supercruise_entry"

    def __init__(self, timeout_seconds: float, poll_interval_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def run(self, context: Context) -> Result:
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            state = context.state_reader.snapshot()
            if state.is_supercruise:
                context.logger.info("Supercruise detected", extra={"state": state.to_debug_dict()})
                return Result.ok("Ship entered supercruise.", debug=state.to_debug_dict())
            time.sleep(self._poll_interval_seconds)

        return Result.fail(
            "Timed out waiting for supercruise entry.",
            debug={"timeout_seconds": self._timeout_seconds},
        )


class WaitForFsdDisengage:
    """Poll state until the ship leaves supercruise after an FSD engagement."""

    name = "wait_for_fsd_disengage"

    def __init__(self, timeout_seconds: float, poll_interval_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def run(self, context: Context) -> Result:
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            state = context.state_reader.snapshot()
            if not state.is_supercruise:
                context.logger.info("FSD disengage detected", extra={"state": state.to_debug_dict()})
                return Result.ok("Ship left supercruise.", debug=state.to_debug_dict())
            time.sleep(self._poll_interval_seconds)

        return Result.fail(
            "Timed out waiting for FSD to disengage.",
            debug={"timeout_seconds": self._timeout_seconds},
        )


def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    context = build_standalone_context(config)

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(f"Warning: focusing game window. Starting {STANDALONE_ACTION} in {STANDALONE_START_DELAY_SECONDS:.1f} seconds...")
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    timings = FsdTimings(
        post_menu_open_wait_seconds=STANDALONE_POST_MENU_OPEN_WAIT_SECONDS,
        post_select_wait_seconds=STANDALONE_POST_SELECT_WAIT_SECONDS,
        key_interval_seconds=STANDALONE_KEY_INTERVAL_SECONDS,
        post_back_wait_seconds=STANDALONE_POST_BACK_WAIT_SECONDS,
        supercruise_entry_timeout_seconds=STANDALONE_SUPERCRUISE_ENTRY_TIMEOUT_SECONDS,
        poll_interval_seconds=STANDALONE_POLL_INTERVAL_SECONDS,
        safety_timeout_seconds=STANDALONE_SAFETY_TIMEOUT_SECONDS,
        cancel_wait_seconds=STANDALONE_CANCEL_WAIT_SECONDS,
    )
    action_map = {
        EngageFsdSequence.name: EngageFsdSequence(timings=timings),
        WaitForMassLockClear.name: WaitForMassLockClear(
            timeout_seconds=timings.supercruise_entry_timeout_seconds,
            poll_interval_seconds=timings.poll_interval_seconds,
        ),
        WaitForSupercruiseEntry.name: WaitForSupercruiseEntry(
            timeout_seconds=timings.supercruise_entry_timeout_seconds,
            poll_interval_seconds=timings.poll_interval_seconds,
        ),
        WaitForFsdDisengage.name: WaitForFsdDisengage(
            timeout_seconds=timings.safety_timeout_seconds,
            poll_interval_seconds=timings.poll_interval_seconds,
        ),
    }
    action = action_map.get(STANDALONE_ACTION)
    if action is None:
        print(f"Unknown standalone fsd action: {STANDALONE_ACTION}")
        return 2
    result = action.run(context)
    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
