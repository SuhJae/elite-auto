from __future__ import annotations

import time
from dataclasses import dataclass

import pydirectinput

from app.domain.protocols import Input, ShipControl


@dataclass(slots=True)
class ShipKeyMap:
    """Central keyboard mapping for ship control commands.

    TODO: Populate this from `.binds` parsing once that adapter exists.
    """

    throttle_zero: str
    throttle_fifty: str
    throttle_seventy_five: str
    throttle_full: str
    throttle_reverse_full: str
    boost: str
    open_left_panel: str
    cycle_previous_panel: str
    cycle_next_panel: str
    ui_up: str
    ui_down: str
    ui_left: str
    ui_right: str
    ui_select: str
    charge_fsd: str


class PydirectInputAdapter(Input):
    """Thin wrapper around pydirectinput for DirectX-friendly keyboard input."""

    def __init__(self, pause_seconds: float = 0.0, press_hold_seconds: float = 0.12) -> None:
        pydirectinput.PAUSE = pause_seconds
        self._press_hold_seconds = press_hold_seconds

    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        for press_index in range(presses):
            self.key_down(key)
            time.sleep(self._press_hold_seconds)
            self.key_up(key)
            if press_index < presses - 1 and interval > 0:
                time.sleep(interval)

    def key_down(self, key: str) -> None:
        pydirectinput.keyDown(key)

    def key_up(self, key: str) -> None:
        pydirectinput.keyUp(key)

    def hold(self, key: str, seconds: float) -> None:
        self.key_down(key)
        time.sleep(seconds)
        self.key_up(key)


class PydirectInputShipControl(ShipControl):
    """Ship-level control surface used by actions and routines."""

    def __init__(self, input_adapter: Input, keymap: ShipKeyMap) -> None:
        self._input = input_adapter
        self._keymap = keymap

    def set_throttle_percent(self, percent: int) -> None:
        throttle_key_by_percent = {
            -100: self._keymap.throttle_reverse_full,
            0: self._keymap.throttle_zero,
            50: self._keymap.throttle_fifty,
            75: self._keymap.throttle_seventy_five,
            100: self._keymap.throttle_full,
        }
        if percent not in throttle_key_by_percent:
            raise ValueError(f"Unsupported throttle preset: {percent}")
        self._input.press(throttle_key_by_percent[percent])

    def boost(self) -> None:
        self._input.press(self._keymap.boost)

    def open_left_panel(self) -> None:
        self._input.press(self._keymap.open_left_panel)

    def cycle_previous_panel(self) -> None:
        self._input.press(self._keymap.cycle_previous_panel)

    def cycle_next_panel(self) -> None:
        self._input.press(self._keymap.cycle_next_panel)

    def ui_select(self, direction: str = "select") -> None:
        direction_map = {
            "up": self._keymap.ui_up,
            "down": self._keymap.ui_down,
            "left": self._keymap.ui_left,
            "right": self._keymap.ui_right,
            "select": self._keymap.ui_select,
        }
        if direction not in direction_map:
            raise ValueError(f"Unsupported UI direction: {direction}")
        self._input.press(direction_map[direction])

    def charge_fsd(self, target: str = "supercruise") -> None:
        # TODO: Different jump targets may need distinct binds or hold timings.
        if target not in {"supercruise", "jump"}:
            raise ValueError(f"Unsupported FSD target: {target}")
        self._input.press(self._keymap.charge_fsd)
