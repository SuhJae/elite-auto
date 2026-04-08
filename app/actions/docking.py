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
STANDALONE_ACTION = "request_docking_sequence"
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_PRE_BOOST_WAIT_SECONDS = 5.0
STANDALONE_POST_BOOST_WAIT_SECONDS = 15.0
STANDALONE_KEY_INTERVAL_SECONDS = 0.5
STANDALONE_POST_SELECT_BACK_WAIT_SECONDS = 2.0
STANDALONE_LANDING_CONFIRM_TIMEOUT_SECONDS = 600.0
STANDALONE_POST_DOCKED_WAIT_SECONDS = 5.0
STANDALONE_POLL_INTERVAL_SECONDS = 1.0


@dataclass(slots=True)
class DockingTimings:
    """Named timings for the docking-request sequence."""

    pre_boost_wait_seconds: float = 5.0
    post_boost_wait_seconds: float = 15.0
    key_interval_seconds: float = 0.5
    post_select_back_wait_seconds: float = 2.0
    landing_confirm_timeout_seconds: float = 600.0
    post_docked_wait_seconds: float = 5.0
    poll_interval_seconds: float = 1.0


@dataclass(slots=True)
class RequestDockingSequence:
    """Wait, boost, then open the left panel and request docking via menu navigation."""

    timings: DockingTimings = field(default_factory=DockingTimings)

    name = "request_docking_sequence"

    def run(self, context: Context) -> Result:
        ship_control = context.ship_control
        input_adapter = context.input_adapter
        if ship_control is None or input_adapter is None:
            return Result.fail("Docking sequence requires both ship control and input control.")

        initial_state = context.state_reader.snapshot()
        if initial_state.is_docked:
            return Result.fail(
                "Cannot run the docking sequence while already docked.",
                debug={"state": initial_state.to_debug_dict()},
            )

        context.logger.info(
            "Starting docking sequence",
            extra={
                "pre_boost_wait_seconds": self.timings.pre_boost_wait_seconds,
                "post_boost_wait_seconds": self.timings.post_boost_wait_seconds,
                "key_interval_seconds": self.timings.key_interval_seconds,
                "post_select_back_wait_seconds": self.timings.post_select_back_wait_seconds,
                "landing_confirm_timeout_seconds": self.timings.landing_confirm_timeout_seconds,
                "post_docked_wait_seconds": self.timings.post_docked_wait_seconds,
            },
        )

        time.sleep(self.timings.pre_boost_wait_seconds)
        ship_control.boost()
        time.sleep(self.timings.post_boost_wait_seconds)

        ship_control.open_left_panel()
        time.sleep(self.timings.key_interval_seconds)
        ship_control.cycle_next_panel()
        time.sleep(self.timings.key_interval_seconds)
        ship_control.cycle_next_panel()
        time.sleep(self.timings.key_interval_seconds)
        ship_control.ui_select("right")
        time.sleep(self.timings.key_interval_seconds)
        ship_control.ui_select("select")
        time.sleep(self.timings.post_select_back_wait_seconds)
        input_adapter.press(context.config.controls.ui_back)

        landing_result = WaitUntilDocked(
            timeout_seconds=self.timings.landing_confirm_timeout_seconds,
            poll_interval_seconds=self.timings.poll_interval_seconds,
        ).run(context)
        if not landing_result.success:
            return Result.fail(
                "Docking request was sent, but landing was not confirmed in time.",
                debug=landing_result.debug,
            )

        time.sleep(self.timings.post_docked_wait_seconds)
        final_state = context.state_reader.snapshot()
        return Result.ok(
            "Docking request sequence completed and landing was confirmed.",
            debug={
                "initial_state": initial_state.to_debug_dict(),
                "final_state": final_state.to_debug_dict(),
                "landing_confirmation": landing_result.debug,
                "pre_boost_wait_seconds": self.timings.pre_boost_wait_seconds,
                "post_boost_wait_seconds": self.timings.post_boost_wait_seconds,
                "tab_right_presses": 2,
                "menu_right_presses": 1,
                "post_select_back_wait_seconds": self.timings.post_select_back_wait_seconds,
                "post_docked_wait_seconds": self.timings.post_docked_wait_seconds,
            },
        )


class WaitUntilDocked:
    """Poll state until the ship is confirmed docked."""

    name = "wait_until_docked"

    def __init__(self, timeout_seconds: float, poll_interval_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def run(self, context: Context) -> Result:
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            state = context.state_reader.snapshot()
            if state.is_docked:
                context.logger.info("Docked state detected", extra={"state": state.to_debug_dict()})
                return Result.ok("Ship is docked.", debug=state.to_debug_dict())
            time.sleep(self._poll_interval_seconds)

        return Result.fail(
            "Timed out waiting for landing confirmation.",
            debug={"timeout_seconds": self._timeout_seconds},
        )


def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    context = build_standalone_context(config)

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(f"Warning: focusing game window. Starting {STANDALONE_ACTION} in {STANDALONE_START_DELAY_SECONDS:.1f} seconds...")
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    timings = DockingTimings(
        pre_boost_wait_seconds=STANDALONE_PRE_BOOST_WAIT_SECONDS,
        post_boost_wait_seconds=STANDALONE_POST_BOOST_WAIT_SECONDS,
        key_interval_seconds=STANDALONE_KEY_INTERVAL_SECONDS,
        post_select_back_wait_seconds=STANDALONE_POST_SELECT_BACK_WAIT_SECONDS,
        landing_confirm_timeout_seconds=STANDALONE_LANDING_CONFIRM_TIMEOUT_SECONDS,
        post_docked_wait_seconds=STANDALONE_POST_DOCKED_WAIT_SECONDS,
        poll_interval_seconds=STANDALONE_POLL_INTERVAL_SECONDS,
    )
    action_map = {
        RequestDockingSequence.name: RequestDockingSequence(timings=timings),
        WaitUntilDocked.name: WaitUntilDocked(
            timeout_seconds=timings.landing_confirm_timeout_seconds,
            poll_interval_seconds=timings.poll_interval_seconds,
        ),
    }
    action = action_map.get(STANDALONE_ACTION)
    if action is None:
        print(f"Unknown standalone docking action: {STANDALONE_ACTION}")
        return 2
    result = action.run(context)
    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
