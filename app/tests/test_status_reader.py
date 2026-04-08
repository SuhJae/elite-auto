from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.state.cargo_reader import CargoReader
from app.state.status_reader import EliteStateReader, StatusFileReader


class TestStatusReader(unittest.TestCase):
    def test_valid_status_maps_flags_and_cargo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            status_path = root / "Status.json"
            cargo_path = root / "Cargo.json"

            status_path.write_text(json.dumps({"Flags": 1 | 16 | 65536, "GuiFocus": 3}), encoding="utf-8")
            cargo_path.write_text(
                json.dumps(
                    {
                        "Inventory": [
                            {"Name": "gold", "Count": 2},
                            {"Name": "silver", "Count": 5},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            reader = EliteStateReader(StatusFileReader(status_path), CargoReader(cargo_path))
            state = reader.snapshot()

            self.assertTrue(state.is_docked)
            self.assertTrue(state.is_supercruise)
            self.assertTrue(state.is_mass_locked)
            self.assertEqual(state.gui_focus, 3)
            self.assertEqual(state.cargo_count, 7)

    def test_missing_status_file_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "Status.json"
            reader = EliteStateReader(StatusFileReader(missing_path))

            with self.assertRaises(FileNotFoundError):
                reader.snapshot()

    def test_malformed_status_json_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "Status.json"
            status_path.write_text("{not valid json", encoding="utf-8")
            reader = EliteStateReader(StatusFileReader(status_path))

            with self.assertRaises(ValueError):
                reader.snapshot()

    def test_retries_transient_invalid_status_json_then_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "Status.json"
            status_path.write_text(json.dumps({"Flags": 16, "GuiFocus": 0}), encoding="utf-8")
            status_reader = StatusFileReader(status_path, retry_attempts=2, retry_interval_seconds=0.0)

            with patch.object(
                status_reader,
                "_read_payload_once",
                side_effect=[
                    ValueError(f"Status file contains invalid JSON: {status_path}"),
                    {"Flags": 16, "GuiFocus": 0},
                ],
            ):
                state = EliteStateReader(status_reader).snapshot()

            self.assertTrue(state.is_supercruise)
            self.assertEqual(state.gui_focus, 0)


if __name__ == "__main__":
    unittest.main()
