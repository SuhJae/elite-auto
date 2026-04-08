from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.actions.fsd import WaitForMassLockClear
from app.actions.starport_buy import build_standalone_context
from app.config import AppConfig
from app.domain.context import Context
from app.domain.result import Result


# Edit these values for standalone testing of this file.
STANDALONE_CONFIG_PATH: str | None = None
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_AUTO_LAUNCH_WAIT_SECONDS = 90.0
STANDALONE_MASS_LOCK_POLL_INTERVAL_SECONDS = 1.0
STANDALONE_POST_MASS_LOCK_CLEAR_WAIT_SECONDS = 20.0
STANDALONE_MASS_LOCK_TIMEOUT_SECONDS = 600.0


@dataclass(slots=True)
class LeaveStationTimings:
    """Named timings for the leave-station sequence."""

    auto_launch_wait_seconds: float = 90.0
    mass_lock_poll_interval_seconds: float = 1.0
    post_mass_lock_clear_wait_seconds: float = 20.0
    mass_lock_timeout_seconds: float = 600.0


@dataclass(slots=True)
class LeaveStation:
    """Activate auto-launch, wait for departure, then clear mass lock."""

    timings: LeaveStationTimings = field(default_factory=LeaveStationTimings)

    name = "leave_station"

    def run(self, context: Context) -> Result:
        ship_control = context.ship_control
        input_adapter = context.input_adapter
        if ship_control is None or input_adapter is None:
            return Result.fail("Leave station requires both ship control and input control.")

        initial_state = context.state_reader.snapshot()
        if not initial_state.is_docked:
            return Result.fail(
                "Cannot run leave_station while already undocked.",
                debug={"state": initial_state.to_debug_dict()},
            )

        context.logger.info("Activating auto-launch")
        ship_control.ui_select("select")

        context.logger.info(
            "Waiting for auto-launch to finish",
            extra={"seconds": self.timings.auto_launch_wait_seconds},
        )
        time.sleep(self.timings.auto_launch_wait_seconds)

        launched_state = context.state_reader.snapshot()
        if launched_state.is_docked:
            return Result.fail(
                "Ship is still docked after the auto-launch wait.",
                debug={"state": launched_state.to_debug_dict()},
            )

        context.logger.info(
            "Holding upward lateral thrust while clearing mass lock",
            extra={
                "thrust_up_binding": context.config.controls.thrust_up,
                "mass_lock_timeout_seconds": self.timings.mass_lock_timeout_seconds,
            },
        )
        input_adapter.key_down(context.config.controls.thrust_up)
        try:
            mass_lock_result = WaitForMassLockClear(
                timeout_seconds=self.timings.mass_lock_timeout_seconds,
                poll_interval_seconds=self.timings.mass_lock_poll_interval_seconds,
            ).run(context)
        finally:
            input_adapter.key_up(context.config.controls.thrust_up)
        if not mass_lock_result.success:
            return Result.fail(
                "Leave station failed while waiting for mass lock to clear.",
                debug=mass_lock_result.debug,
            )

        time.sleep(self.timings.post_mass_lock_clear_wait_seconds)
        ship_control.set_throttle_percent(0)

        final_state = context.state_reader.snapshot()
        return Result.ok(
            "Leave-station sequence completed.",
            debug={
                "mass_lock_clear": mass_lock_result.debug,
                "mass_locked": final_state.is_mass_locked,
                "is_docked": final_state.is_docked,
                "thrust_up_binding": context.config.controls.thrust_up,
                "post_mass_lock_clear_wait_seconds": self.timings.post_mass_lock_clear_wait_seconds,
            },
        )


def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    context = build_standalone_context(config)

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(f"Warning: focusing game window. Starting leave_station in {STANDALONE_START_DELAY_SECONDS:.1f} seconds...")
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    action = LeaveStation(
        timings=LeaveStationTimings(
            auto_launch_wait_seconds=STANDALONE_AUTO_LAUNCH_WAIT_SECONDS,
            mass_lock_poll_interval_seconds=STANDALONE_MASS_LOCK_POLL_INTERVAL_SECONDS,
            post_mass_lock_clear_wait_seconds=STANDALONE_POST_MASS_LOCK_CLEAR_WAIT_SECONDS,
            mass_lock_timeout_seconds=STANDALONE_MASS_LOCK_TIMEOUT_SECONDS,
        )
    )
    result = action.run(context)
    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
