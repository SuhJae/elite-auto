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
STANDALONE_ACTION = "unload_construction"
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_MENU_OPEN_WAIT_SECONDS = 5.0
STANDALONE_SCROLL_DOWN_HOLD_SECONDS = 10.0
STANDALONE_KEY_INTERVAL_SECONDS = 0.5
STANDALONE_POST_CONFIRM_WAIT_SECONDS = 5.0


@dataclass(slots=True)
class UnloadConstructionTimings:
    """Named timings for the construction unload sequence."""

    menu_open_wait_seconds: float = 5.0
    scroll_down_hold_seconds: float = 10.0
    key_interval_seconds: float = 0.5
    post_confirm_wait_seconds: float = 5.0


@dataclass(slots=True)
class UnloadConstruction:
    """Open the construction menu and perform the unload input sequence."""

    timings: UnloadConstructionTimings = field(default_factory=UnloadConstructionTimings)

    name = "unload_construction"

    def run(self, context: Context) -> Result:
        ship_control = context.ship_control
        input_adapter = context.input_adapter
        if ship_control is None or input_adapter is None:
            return Result.fail("Unload construction requires both ship control and input control.")

        initial_state = context.state_reader.snapshot()
        if not initial_state.is_docked:
            return Result.fail(
                "Cannot unload construction cargo while undocked.",
                debug={"state": initial_state.to_debug_dict()},
            )

        context.logger.info(
            "Starting unload construction sequence",
            extra={
                "menu_open_wait_seconds": self.timings.menu_open_wait_seconds,
                "scroll_down_hold_seconds": self.timings.scroll_down_hold_seconds,
                "key_interval_seconds": self.timings.key_interval_seconds,
                "post_confirm_wait_seconds": self.timings.post_confirm_wait_seconds,
            },
        )

        ship_control.ui_select("select")
        time.sleep(self.timings.menu_open_wait_seconds)

        input_adapter.hold(context.config.controls.ui_down, self.timings.scroll_down_hold_seconds)
        time.sleep(self.timings.key_interval_seconds)
        ship_control.ui_select("up")
        time.sleep(self.timings.key_interval_seconds)
        ship_control.ui_select("right")
        time.sleep(self.timings.key_interval_seconds)
        ship_control.ui_select("right")
        time.sleep(self.timings.key_interval_seconds)
        ship_control.ui_select("select")
        time.sleep(self.timings.key_interval_seconds)
        ship_control.ui_select("left")
        time.sleep(self.timings.key_interval_seconds)
        ship_control.ui_select("select")
        time.sleep(self.timings.post_confirm_wait_seconds)
        input_adapter.press(context.config.controls.ui_back)
        time.sleep(self.timings.key_interval_seconds)
        ship_control.ui_select("down")

        final_state = context.state_reader.snapshot()
        return Result.ok(
            "Unload construction sequence completed.",
            debug={
                "initial_state": initial_state.to_debug_dict(),
                "final_state": final_state.to_debug_dict(),
                "menu_open_wait_seconds": self.timings.menu_open_wait_seconds,
                "scroll_down_hold_seconds": self.timings.scroll_down_hold_seconds,
                "post_confirm_wait_seconds": self.timings.post_confirm_wait_seconds,
            },
        )


def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    context = build_standalone_context(config)

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(f"Warning: focusing game window. Starting {STANDALONE_ACTION} in {STANDALONE_START_DELAY_SECONDS:.1f} seconds...")
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    timings = UnloadConstructionTimings(
        menu_open_wait_seconds=STANDALONE_MENU_OPEN_WAIT_SECONDS,
        scroll_down_hold_seconds=STANDALONE_SCROLL_DOWN_HOLD_SECONDS,
        key_interval_seconds=STANDALONE_KEY_INTERVAL_SECONDS,
        post_confirm_wait_seconds=STANDALONE_POST_CONFIRM_WAIT_SECONDS,
    )
    action_map = {
        UnloadConstruction.name: UnloadConstruction(timings=timings),
    }
    action = action_map.get(STANDALONE_ACTION)
    if action is None:
        print(f"Unknown standalone unload_construction action: {STANDALONE_ACTION}")
        return 2
    result = action.run(context)
    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
