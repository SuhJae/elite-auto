from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AppConfig
from app.domain.context import Context
from app.domain.result import Result

try:
    from app.actions.navigation_ocr import MoveCursorToNavTarget, OcrNavConfig, OcrNavTimings
except ModuleNotFoundError:
    class MoveCursorToNavTarget:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            return None

        def run(self, context):
            return Result.fail("Navigation OCR dependencies are not available in the current environment.")

    @dataclass(slots=True)
    class OcrNavTimings:  # type: ignore[no-redef]
        move_interval_seconds: float = 0.2

    @dataclass(slots=True)
    class OcrNavConfig:  # type: ignore[no-redef]
        save_debug_artifacts: bool = False


# Edit these values for standalone testing of this file.
STANDALONE_CONFIG_PATH: str | None = None
STANDALONE_ACTION = "lock_nav_destination"
# STANDALONE_TARGET_NAME = "ADENAUER SANCTUARY"
STANDALONE_TARGET_NAME = "Khun Port"
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_PANEL_OPEN_CONFIRM_TIMEOUT_SECONDS = 2.0
STANDALONE_PANEL_OPEN_RETRY_COUNT = 3
STANDALONE_PANEL_OPEN_POLL_INTERVAL_SECONDS = 0.2
STANDALONE_TAB_PREVIOUS_INTERVAL_SECONDS = 0.5
STANDALONE_ANCHOR_HOLD_SECONDS = 10.0
STANDALONE_ANCHOR_SETTLE_SECONDS = 0.5
STANDALONE_LOCK_SELECT_INTERVAL_SECONDS = 0.5
STANDALONE_BACK_TO_COCKPIT_WAIT_SECONDS = 0.5
STANDALONE_SAVE_DEBUG_ARTIFACTS = True

LEFT_PANEL_GUI_FOCUS = 2


@dataclass(slots=True)
class NavigationTimings:
    """Named timings for small navigation-panel actions."""

    panel_open_confirm_timeout_seconds: float = 2.0
    panel_open_retry_count: int = 3
    panel_open_poll_interval_seconds: float = 0.2
    tab_previous_interval_seconds: float = 0.5
    anchor_hold_seconds: float = 10.0
    anchor_settle_seconds: float = 0.5
    lock_select_interval_seconds: float = 0.5
    back_to_cockpit_wait_seconds: float = 0.5


@dataclass(slots=True)
class OpenNavPanel:
    """Open the left navigation panel and anchor the list at the top."""

    timings: NavigationTimings = field(default_factory=NavigationTimings)

    name = "open_nav_panel"

    def run(self, context: Context) -> Result:
        ship_control = context.ship_control
        input_adapter = context.input_adapter
        if ship_control is None or input_adapter is None:
            return Result.fail("Input control is not available in the current context.")

        context.logger.info("Opening left navigation panel")
        if not self._ensure_left_panel_open(context):
            state = context.state_reader.snapshot()
            return Result.fail(
                "Failed to confirm left navigation panel open from Status.json.",
                debug={
                    "gui_focus": state.gui_focus,
                    "expected_gui_focus": LEFT_PANEL_GUI_FOCUS,
                    "panel_open_retry_count": self.timings.panel_open_retry_count,
                    "panel_open_confirm_timeout_seconds": self.timings.panel_open_confirm_timeout_seconds,
                },
            )
        ship_control.cycle_previous_panel()
        time.sleep(self.timings.tab_previous_interval_seconds)
        ship_control.cycle_previous_panel()
        time.sleep(self.timings.tab_previous_interval_seconds)
        input_adapter.hold(context.config.controls.ui_up, self.timings.anchor_hold_seconds)
        time.sleep(self.timings.anchor_settle_seconds)
        return Result.ok(
            "Navigation panel opened and anchored to the top of the list.",
            debug={
                "gui_focus": LEFT_PANEL_GUI_FOCUS,
                "panel_open_retry_count": self.timings.panel_open_retry_count,
                "panel_open_confirm_timeout_seconds": self.timings.panel_open_confirm_timeout_seconds,
                "tab_previous_presses": 2,
                "tab_previous_interval_seconds": self.timings.tab_previous_interval_seconds,
                "anchor_hold_seconds": self.timings.anchor_hold_seconds,
                "anchor_settle_seconds": self.timings.anchor_settle_seconds,
            },
        )

    def _ensure_left_panel_open(self, context: Context) -> bool:
        ship_control = context.ship_control
        if ship_control is None:
            raise RuntimeError("Ship control is not available.")

        for attempt in range(1, self.timings.panel_open_retry_count + 1):
            ship_control.open_left_panel()
            if self._wait_for_gui_focus(
                context=context,
                expected_gui_focus=LEFT_PANEL_GUI_FOCUS,
                timeout_seconds=self.timings.panel_open_confirm_timeout_seconds,
                poll_interval_seconds=self.timings.panel_open_poll_interval_seconds,
            ):
                context.logger.info(
                    "Left navigation panel confirmed open",
                    extra={"attempt": attempt, "expected_gui_focus": LEFT_PANEL_GUI_FOCUS},
                )
                return True

            context.logger.warning(
                "Left navigation panel did not confirm open; retrying",
                extra={"attempt": attempt, "expected_gui_focus": LEFT_PANEL_GUI_FOCUS},
            )

        return False

    @staticmethod
    def _wait_for_gui_focus(
        context: Context,
        expected_gui_focus: int,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            state = context.state_reader.snapshot()
            if state.gui_focus == expected_gui_focus:
                return True
            time.sleep(poll_interval_seconds)
        return False


@dataclass(slots=True)
class LockNavDestination:
    """Open nav, OCR-select a destination, lock it, then return to cockpit."""

    target_name: str
    timings: NavigationTimings = field(default_factory=NavigationTimings)
    ocr_config: OcrNavConfig = field(default_factory=OcrNavConfig)

    name = "lock_nav_destination"

    def run(self, context: Context) -> Result:
        ship_control = context.ship_control
        input_adapter = context.input_adapter
        if ship_control is None or input_adapter is None:
            return Result.fail("Input control is not available in the current context.")

        open_result = OpenNavPanel(timings=self.timings).run(context)
        if not open_result.success:
            return Result.fail(
                "Failed to initialize navigation panel before OCR target selection.",
                debug=open_result.debug,
            )

        ocr_result = MoveCursorToNavTarget(
            target_name=self.target_name,
            timings=OcrNavTimings(move_interval_seconds=0.2),
            config=self.ocr_config,
        ).run(context)
        if not ocr_result.success:
            return Result.fail(
                "Failed to move navigation cursor onto the requested target.",
                debug=ocr_result.debug,
            )

        ship_control.ui_select("select")
        time.sleep(self.timings.lock_select_interval_seconds)
        ship_control.ui_select("select")
        time.sleep(self.timings.lock_select_interval_seconds)
        input_adapter.press(context.config.controls.ui_back)
        time.sleep(self.timings.back_to_cockpit_wait_seconds)

        return Result.ok(
            "Navigation destination locked and cockpit restored.",
            debug={
                "target_name": self.target_name,
                "lock_select_interval_seconds": self.timings.lock_select_interval_seconds,
                "back_to_cockpit_wait_seconds": self.timings.back_to_cockpit_wait_seconds,
                "open_nav_panel": open_result.debug,
                "ocr_targeting": ocr_result.debug,
            },
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


def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    from app.actions.starport_buy import build_standalone_context

    context = build_standalone_context(config)

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(f"Warning: focusing game window. Starting {STANDALONE_ACTION} in {STANDALONE_START_DELAY_SECONDS:.1f} seconds...")
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    timings = NavigationTimings(
        panel_open_confirm_timeout_seconds=STANDALONE_PANEL_OPEN_CONFIRM_TIMEOUT_SECONDS,
        panel_open_retry_count=STANDALONE_PANEL_OPEN_RETRY_COUNT,
        panel_open_poll_interval_seconds=STANDALONE_PANEL_OPEN_POLL_INTERVAL_SECONDS,
        tab_previous_interval_seconds=STANDALONE_TAB_PREVIOUS_INTERVAL_SECONDS,
        anchor_hold_seconds=STANDALONE_ANCHOR_HOLD_SECONDS,
        anchor_settle_seconds=STANDALONE_ANCHOR_SETTLE_SECONDS,
        lock_select_interval_seconds=STANDALONE_LOCK_SELECT_INTERVAL_SECONDS,
        back_to_cockpit_wait_seconds=STANDALONE_BACK_TO_COCKPIT_WAIT_SECONDS,
    )
    action_map = {
        OpenNavPanel.name: OpenNavPanel(timings=timings),
        LockNavDestination.name: LockNavDestination(
            target_name=STANDALONE_TARGET_NAME,
            timings=timings,
            ocr_config=OcrNavConfig(save_debug_artifacts=STANDALONE_SAVE_DEBUG_ARTIFACTS),
        ),
    }
    action = action_map.get(STANDALONE_ACTION)
    if action is None:
        print(f"Unknown standalone navigation action: {STANDALONE_ACTION}")
        return 2
    result = action.run(context)
    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
