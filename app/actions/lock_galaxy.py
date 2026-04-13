from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from ctypes import wintypes

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import AppConfig
from app.domain.context import Context
from app.domain.result import Result


STANDALONE_CONFIG_PATH: str | None = None
STANDALONE_TARGET_NAME = "BODEDI"
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_WINDOW_TITLE = "Elite Dangerous"
STANDALONE_OPEN_KEY = "f4"
STANDALONE_MAP_OPEN_TIMEOUT_SECONDS = 5.0
STANDALONE_MAP_POLL_INTERVAL_SECONDS = 0.2
STANDALONE_MAP_RETRY_COUNT = 2
STANDALONE_POST_MAP_OPEN_DELAY_SECONDS = 1.0
STANDALONE_KEY_INTERVAL_SECONDS = 0.2
STANDALONE_SEARCH_SETTLE_SECONDS = 5.0
STANDALONE_POST_RESULT_SELECT_DELAY_SECONDS = 2.0
STANDALONE_CLICK_HOLD_SECONDS = 2.5
STANDALONE_DESTINATION_CONFIRM_TIMEOUT_SECONDS = 5.0
STANDALONE_CLOSE_TIMEOUT_SECONDS = 5.0
STANDALONE_CLOSE_DELETE_PRESSES = 2
STANDALONE_CLOSE_EXTRA_DELETE_PRESSES = 6

GALAXY_MAP_GUI_FOCUS = 6


@dataclass(slots=True)
class GalaxyLockTimings:
    """Named timings for galaxy-map destination locking."""

    map_open_timeout_seconds: float = 5.0
    map_poll_interval_seconds: float = 0.2
    map_retry_count: int = 2
    post_map_open_delay_seconds: float = 1.0
    key_interval_seconds: float = 0.2
    search_settle_seconds: float = 5.0
    post_result_select_delay_seconds: float = 2.0
    click_hold_seconds: float = 2.5
    destination_confirm_timeout_seconds: float = 5.0
    close_timeout_seconds: float = 5.0
    close_delete_presses: int = 2
    close_extra_delete_presses: int = 6


@dataclass(slots=True)
class GalaxyLockConfig:
    """Configuration for galaxy-map locking helpers."""

    window_title_substring: str = "Elite Dangerous"
    open_key: str = "f4"


@dataclass(slots=True)
class LockGalaxy:
    """Open the galaxy map, search a destination, and lock it using Status.json confirmation."""

    target_name: str
    timings: GalaxyLockTimings = field(default_factory=GalaxyLockTimings)
    config: GalaxyLockConfig = field(default_factory=GalaxyLockConfig)

    name = "lock_galaxy"

    def run(self, context: Context) -> Result:
        ship_control = context.ship_control
        input_adapter = context.input_adapter
        if ship_control is None or input_adapter is None:
            return Result.fail("Input control is not available in the current context.")

        target_key = _normalize_galaxy_target(self.target_name)
        if not target_key:
            return Result.fail("A target galaxy destination is required.", debug={"target_name": self.target_name})

        initial_state = context.state_reader.snapshot()
        initial_destination = _destination_name_from_state(initial_state)

        for attempt in range(1, self.timings.map_retry_count + 1):
            context.logger.info(
                "Starting galaxy lock attempt",
                extra={"attempt": attempt, "target_name": self.target_name, "initial_destination": initial_destination},
            )

            input_adapter.press(self.config.open_key)
            if not _wait_for_gui_focus(
                context=context,
                expected_gui_focus=GALAXY_MAP_GUI_FOCUS,
                timeout_seconds=self.timings.map_open_timeout_seconds,
                poll_interval_seconds=self.timings.map_poll_interval_seconds,
            ):
                _close_galaxy_map(context, self.timings)
                continue

            time.sleep(self.timings.post_map_open_delay_seconds)
            input_adapter.press(context.config.controls.ui_up)
            time.sleep(self.timings.key_interval_seconds)
            ship_control.ui_select("select")
            time.sleep(self.timings.key_interval_seconds)

            _type_galaxy_target(input_adapter, self.target_name, self.timings.key_interval_seconds)
            time.sleep(self.timings.search_settle_seconds)

            ship_control.ui_select("down")
            time.sleep(self.timings.key_interval_seconds)
            ship_control.ui_select("select")
            time.sleep(self.timings.post_result_select_delay_seconds)

            if not _move_mouse_to_window_center_and_hold(
                window_title_substring=self.config.window_title_substring,
                hold_seconds=self.timings.click_hold_seconds,
            ):
                _close_galaxy_map(context, self.timings)
                continue

            if not _wait_for_destination_name(
                context=context,
                expected_name=self.target_name,
                timeout_seconds=self.timings.destination_confirm_timeout_seconds,
                poll_interval_seconds=self.timings.map_poll_interval_seconds,
            ):
                _close_galaxy_map(context, self.timings)
                continue

            if not _close_galaxy_map(context, self.timings):
                return Result.fail(
                    "Galaxy destination updated but the galaxy map did not close cleanly.",
                    debug={
                        "target_name": self.target_name,
                        "attempt": attempt,
                        "gui_focus": context.state_reader.snapshot().gui_focus,
                    },
                )

            final_state = context.state_reader.snapshot()
            return Result.ok(
                "Galaxy destination locked.",
                debug={
                    "target_name": self.target_name,
                    "attempt": attempt,
                    "destination_before": initial_destination,
                    "destination_after": _destination_name_from_state(final_state),
                },
            )

        _close_galaxy_map(context, self.timings)
        return Result.fail(
            "Failed to lock galaxy destination after retries.",
            debug={"target_name": self.target_name, "destination_before": initial_destination},
        )


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


def _wait_for_destination_name(
    context: Context,
    expected_name: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> bool:
    expected_key = _normalize_galaxy_target(expected_name)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = context.state_reader.snapshot()
        if _normalize_galaxy_target(_destination_name_from_state(state)) == expected_key:
            return True
        time.sleep(poll_interval_seconds)
    return False


def _close_galaxy_map(context: Context, timings: GalaxyLockTimings) -> bool:
    input_adapter = context.input_adapter
    if input_adapter is None:
        raise RuntimeError("Input control is not available.")

    input_adapter.press(context.config.controls.ui_back, presses=timings.close_delete_presses)
    if _wait_for_map_close(context, timings.close_timeout_seconds, timings.map_poll_interval_seconds):
        return True

    for _ in range(timings.close_extra_delete_presses):
        input_adapter.press(context.config.controls.ui_back)
        if _wait_for_map_close(context, timings.map_poll_interval_seconds, timings.map_poll_interval_seconds):
            return True
    return _wait_for_map_close(context, timings.map_poll_interval_seconds, timings.map_poll_interval_seconds)


def _wait_for_map_close(context: Context, timeout_seconds: float, poll_interval_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = context.state_reader.snapshot()
        if state.gui_focus != GALAXY_MAP_GUI_FOCUS:
            return True
        time.sleep(poll_interval_seconds)
    return False


def _type_galaxy_target(input_adapter, target_name: str, key_interval_seconds: float) -> None:
    for key in _galaxy_target_keys(target_name):
        input_adapter.press(key)
        time.sleep(key_interval_seconds)


def _galaxy_target_keys(target_name: str) -> list[str]:
    key_sequence: list[str] = []
    for character in target_name:
        if character.isalpha():
            key_sequence.append(character.lower())
        elif character.isdigit():
            key_sequence.append(character)
        elif character == " ":
            key_sequence.append("space")
        elif character == "-":
            key_sequence.append("minus")
        elif character == "'":
            key_sequence.append("apostrophe")
    return key_sequence


def _destination_name_from_state(state) -> str | None:
    destination = state.raw_status.get("Destination") if isinstance(state.raw_status, dict) else None
    if not isinstance(destination, dict):
        return None
    name = destination.get("Name")
    return str(name) if name is not None else None


def _normalize_galaxy_target(value: str | None) -> str:
    return "".join(character.lower() for character in (value or "") if character.isalnum())


def _find_window_client_region(window_title_substring: str) -> tuple[int, int, int, int] | None:
    user32 = ctypes.windll.user32
    found_hwnd: list[int] = []
    title_match = _normalize_window_title(window_title_substring)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_windows_proc(hwnd: int, lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, len(buffer))
        title = buffer.value.strip()
        if title_match not in _normalize_window_title(title):
            return True

        found_hwnd.append(hwnd)
        return False

    user32.EnumWindows(enum_windows_proc, 0)
    if not found_hwnd:
        return None

    hwnd = found_hwnd[0]
    if user32.IsIconic(hwnd):
        return None

    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None

    top_left = wintypes.POINT(rect.left, rect.top)
    bottom_right = wintypes.POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        return None

    width = bottom_right.x - top_left.x
    height = bottom_right.y - top_left.y
    if width <= 0 or height <= 0:
        return None

    return top_left.x, top_left.y, width, height


def _normalize_window_title(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _move_mouse_to_window_center_and_hold(window_title_substring: str, hold_seconds: float) -> bool:
    region = _find_window_client_region(window_title_substring)
    if region is None:
        return False

    x, y, width, height = region
    center_x = x + (width // 2)
    center_y = y + (height // 2)

    import pydirectinput

    pydirectinput.moveTo(center_x, center_y)
    pydirectinput.mouseDown(button="left")
    time.sleep(hold_seconds)
    pydirectinput.mouseUp(button="left")
    return True


def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    from app.actions.starport_buy import build_standalone_context

    context = build_standalone_context(config)

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(
            f"Warning: focusing game window. Starting {LockGalaxy.name} in {STANDALONE_START_DELAY_SECONDS:.1f} seconds..."
        )
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    action = LockGalaxy(
        target_name=STANDALONE_TARGET_NAME,
        timings=GalaxyLockTimings(
            map_open_timeout_seconds=STANDALONE_MAP_OPEN_TIMEOUT_SECONDS,
            map_poll_interval_seconds=STANDALONE_MAP_POLL_INTERVAL_SECONDS,
            map_retry_count=STANDALONE_MAP_RETRY_COUNT,
            post_map_open_delay_seconds=STANDALONE_POST_MAP_OPEN_DELAY_SECONDS,
            key_interval_seconds=STANDALONE_KEY_INTERVAL_SECONDS,
            search_settle_seconds=STANDALONE_SEARCH_SETTLE_SECONDS,
            post_result_select_delay_seconds=STANDALONE_POST_RESULT_SELECT_DELAY_SECONDS,
            click_hold_seconds=STANDALONE_CLICK_HOLD_SECONDS,
            destination_confirm_timeout_seconds=STANDALONE_DESTINATION_CONFIRM_TIMEOUT_SECONDS,
            close_timeout_seconds=STANDALONE_CLOSE_TIMEOUT_SECONDS,
            close_delete_presses=STANDALONE_CLOSE_DELETE_PRESSES,
            close_extra_delete_presses=STANDALONE_CLOSE_EXTRA_DELETE_PRESSES,
        ),
        config=GalaxyLockConfig(
            window_title_substring=STANDALONE_WINDOW_TITLE,
            open_key=STANDALONE_OPEN_KEY,
        ),
    )
    result = action.run(context)
    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
