from __future__ import annotations

import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from app.config import ControlConfig, PathConfig


SUPPORTED_BINDING_TAGS: dict[str, str] = {
    "SetSpeedZero": "throttle_zero",
    "SetSpeed50": "throttle_fifty",
    "SetSpeed75": "throttle_seventy_five",
    "SetSpeed100": "throttle_full",
    "SetSpeedMinus100": "throttle_reverse_full",
    "UseBoostJuice": "boost",
    "FocusLeftPanel": "open_left_panel",
    "CyclePreviousPanel": "cycle_previous_panel",
    "CycleNextPanel": "cycle_next_panel",
    "UI_Up": "ui_up",
    "UI_Down": "ui_down",
    "UI_Left": "ui_left",
    "UI_Right": "ui_right",
    "UI_Select": "ui_select",
    "UI_Back": "ui_back",
    "HyperSuperCombination": "charge_fsd",
    "UpThrustButton": "thrust_up",
    "PitchUpButton": "pitch_up",
    "PitchDownButton": "pitch_down",
    "YawLeftButton": "yaw_left",
    "YawRightButton": "yaw_right",
}

UI_DIRECTION_BINDING_TAGS = {
    "UI_Up",
    "UI_Down",
    "UI_Left",
    "UI_Right",
}

SPECIAL_KEY_MAP = {
    "Key_Space": "space",
    "Key_Tab": "tab",
    "Key_Backspace": "backspace",
    "Key_UpArrow": "up",
    "Key_DownArrow": "down",
    "Key_LeftArrow": "left",
    "Key_RightArrow": "right",
    "Key_LeftShift": "shiftleft",
    "Key_RightShift": "shiftright",
    "Key_LeftControl": "ctrlleft",
    "Key_RightControl": "ctrlright",
    "Key_LeftAlt": "altleft",
    "Key_RightAlt": "altright",
    "Key_Return": "enter",
    "Key_Enter": "enter",
    "Key_Escape": "esc",
    "Key_Minus": "-",
    "Key_Equals": "=",
    "Key_Semicolon": ";",
    "Key_Apostrophe": "'",
    "Key_Comma": ",",
    "Key_Period": ".",
    "Key_Slash": "/",
    "Key_Backslash": "\\",
    "Key_LeftBracket": "[",
    "Key_RightBracket": "]",
    "Key_Grave": "`",
}


@dataclass(slots=True)
class DetectedBindings:
    """Resolved keyboard bindings from an Elite Dangerous preset."""

    preset_name: str
    binds_path: Path
    controls: ControlConfig
    detected_fields: dict[str, str]


class EliteBindingsReader:
    """Discover and parse the active Elite Dangerous `.binds` preset."""

    def __init__(self, paths: PathConfig, logger: logging.Logger | None = None) -> None:
        self._paths = paths
        self._logger = logger or logging.getLogger(__name__)

    def detect_controls(self, defaults: ControlConfig) -> DetectedBindings | None:
        binds_path = self.find_active_binds_file()
        if binds_path is None:
            self._logger.info(
                "No Elite bindings file detected; using configured control defaults.",
                extra={"bindings_dir": str(self._paths.bindings_dir)},
            )
            return None

        try:
            root = ET.parse(binds_path).getroot()
        except ET.ParseError:
            self._logger.warning(
                "Failed to parse Elite bindings file; using configured control defaults.",
                extra={"binds_path": str(binds_path)},
            )
            return None

        preset_name = root.attrib.get("PresetName", binds_path.stem)
        resolved = {
            "throttle_zero": defaults.throttle_zero,
            "throttle_fifty": defaults.throttle_fifty,
            "throttle_seventy_five": defaults.throttle_seventy_five,
            "throttle_full": defaults.throttle_full,
            "throttle_reverse_full": defaults.throttle_reverse_full,
            "boost": defaults.boost,
            "open_left_panel": defaults.open_left_panel,
            "cycle_previous_panel": defaults.cycle_previous_panel,
            "cycle_next_panel": defaults.cycle_next_panel,
            "ui_up": defaults.ui_up,
            "ui_down": defaults.ui_down,
            "ui_left": defaults.ui_left,
            "ui_right": defaults.ui_right,
            "ui_select": defaults.ui_select,
            "ui_back": defaults.ui_back,
            "charge_fsd": defaults.charge_fsd,
            "thrust_up": defaults.thrust_up,
            "pitch_up": defaults.pitch_up,
            "pitch_down": defaults.pitch_down,
            "yaw_left": defaults.yaw_left,
            "yaw_right": defaults.yaw_right,
        }
        detected_fields: dict[str, str] = {}

        for xml_tag, field_name in SUPPORTED_BINDING_TAGS.items():
            node = root.find(xml_tag)
            if node is None:
                continue
            if xml_tag in UI_DIRECTION_BINDING_TAGS:
                key = self._extract_preferred_ui_direction_key(node)
            else:
                key = self._extract_keyboard_key(node)
            if key is None:
                continue
            resolved[field_name] = key
            detected_fields[field_name] = key

        controls = ControlConfig(**resolved)
        return DetectedBindings(
            preset_name=preset_name,
            binds_path=binds_path,
            controls=controls,
            detected_fields=detected_fields,
        )

    def find_active_binds_file(self) -> Path | None:
        bindings_dir = self._paths.bindings_dir
        if not bindings_dir.exists():
            return None

        preset_name = self._read_start_preset_name()
        if preset_name:
            matching = sorted(bindings_dir.glob(f"{preset_name}*.binds"), key=lambda path: path.stat().st_mtime)
            if matching:
                return matching[-1]

        all_binds = sorted(bindings_dir.glob("*.binds"), key=lambda path: path.stat().st_mtime)
        if all_binds:
            return all_binds[-1]
        return None

    def _read_start_preset_name(self) -> str | None:
        start_file = self._paths.start_preset_file
        if not start_file.exists():
            return None

        lines = [line.strip() for line in start_file.read_text(encoding="utf-8").splitlines()]
        for line in lines:
            if line:
                return line
        return None

    def _extract_keyboard_key(self, node: ET.Element) -> str | None:
        primary = node.find("Primary")
        secondary = node.find("Secondary")
        binding = node.find("Binding")

        for candidate in (primary, secondary, binding):
            if candidate is None:
                continue
            device = candidate.attrib.get("Device", "")
            raw_key = candidate.attrib.get("Key", "")
            converted = _convert_elite_key_to_pydirectinput(device=device, raw_key=raw_key)
            if converted is not None:
                return converted
        return None

    def _extract_preferred_ui_direction_key(self, node: ET.Element) -> str | None:
        primary = node.find("Primary")
        secondary = node.find("Secondary")
        binding = node.find("Binding")

        for candidate in (primary, secondary, binding):
            if candidate is None:
                continue
            converted = _convert_elite_key_to_pydirectinput(
                device=candidate.attrib.get("Device", ""),
                raw_key=candidate.attrib.get("Key", ""),
            )
            if converted in {"up", "down", "left", "right"}:
                return converted

        return self._extract_keyboard_key(node)


def _convert_elite_key_to_pydirectinput(device: str, raw_key: str) -> str | None:
    if device != "Keyboard" or not raw_key:
        return None

    if raw_key in SPECIAL_KEY_MAP:
        return SPECIAL_KEY_MAP[raw_key]

    if raw_key.startswith("Key_") and len(raw_key) == 5:
        return raw_key[-1].lower()

    if raw_key.startswith("Key_") and len(raw_key) == 6 and raw_key[-1].isdigit():
        return raw_key[-1]

    return None
