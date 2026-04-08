from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from app.config import ControlConfig, PathConfig
from app.state.bindings_reader import EliteBindingsReader


class TestBindingsReader(unittest.TestCase):
    def test_detects_active_preset_and_keyboard_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bindings_dir = root / "Bindings"
            bindings_dir.mkdir()
            start_file = bindings_dir / "StartPreset.4.start"
            start_file.write_text("Custom\n", encoding="utf-8")

            binds_path = bindings_dir / "Custom.4.2.binds"
            binds_path.write_text(
                textwrap.dedent(
                    """\
                    <?xml version="1.0" encoding="UTF-8" ?>
                    <Root PresetName="Custom" MajorVersion="4" MinorVersion="2">
                        <SetSpeedZero>
                            <Primary Device="Keyboard" Key="Key_X" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </SetSpeedZero>
                        <SetSpeedMinus100>
                            <Primary Device="Keyboard" Key="Key_9" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </SetSpeedMinus100>
                        <UseBoostJuice>
                            <Primary Device="Keyboard" Key="Key_Tab" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </UseBoostJuice>
                        <FocusLeftPanel>
                            <Primary Device="Keyboard" Key="Key_1" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </FocusLeftPanel>
                        <CycleNextPanel>
                            <Primary Device="Keyboard" Key="Key_E" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </CycleNextPanel>
                        <UI_Up>
                            <Primary Device="Keyboard" Key="Key_W" />
                            <Secondary Device="Keyboard" Key="Key_UpArrow" />
                        </UI_Up>
                        <UI_Select>
                            <Primary Device="Keyboard" Key="Key_Space" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </UI_Select>
                        <HyperSuperCombination>
                            <Primary Device="Keyboard" Key="Key_J" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </HyperSuperCombination>
                        <UpThrustButton>
                            <Primary Device="Keyboard" Key="Key_R" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </UpThrustButton>
                        <PitchUpButton>
                            <Primary Device="Keyboard" Key="Key_Y" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </PitchUpButton>
                        <PitchDownButton>
                            <Primary Device="Keyboard" Key="Key_H" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </PitchDownButton>
                        <YawLeftButton>
                            <Primary Device="Keyboard" Key="Key_A" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </YawLeftButton>
                        <YawRightButton>
                            <Primary Device="Keyboard" Key="Key_D" />
                            <Secondary Device="{NoDevice}" Key="" />
                        </YawRightButton>
                    </Root>
                    """
                ),
                encoding="utf-8",
            )

            paths = PathConfig(bindings_dir=bindings_dir, start_preset_file=start_file)
            detected = EliteBindingsReader(paths).detect_controls(ControlConfig())

            self.assertIsNotNone(detected)
            assert detected is not None
            self.assertEqual(detected.preset_name, "Custom")
            self.assertEqual(detected.binds_path, binds_path)
            self.assertEqual(detected.controls.throttle_zero, "x")
            self.assertEqual(detected.controls.throttle_reverse_full, "9")
            self.assertEqual(detected.controls.boost, "tab")
            self.assertEqual(detected.controls.open_left_panel, "1")
            self.assertEqual(detected.controls.cycle_next_panel, "e")
            self.assertEqual(detected.controls.ui_up, "w")
            self.assertEqual(detected.controls.ui_select, "space")
            self.assertEqual(detected.controls.charge_fsd, "j")
            self.assertEqual(detected.controls.thrust_up, "r")
            self.assertEqual(detected.controls.pitch_up, "y")
            self.assertEqual(detected.controls.pitch_down, "h")
            self.assertEqual(detected.controls.yaw_left, "a")
            self.assertEqual(detected.controls.yaw_right, "d")

    def test_returns_none_when_bindings_dir_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = PathConfig(bindings_dir=root / "Missing", start_preset_file=root / "Missing.start")
            detected = EliteBindingsReader(paths).detect_controls(ControlConfig())
            self.assertIsNone(detected)


if __name__ == "__main__":
    unittest.main()
