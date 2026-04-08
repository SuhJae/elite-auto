from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.state.market_reader import MarketReader


class TestMarketReader(unittest.TestCase):
    def test_snapshot_parses_commodities_and_matches_last_docked_station(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            market_path = root / "Market.json"
            journal_path = root / "Journal.2026-04-07T230000.01.log"

            market_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-04-07T23:19:51Z",
                        "event": "Market",
                        "MarketID": 3700490240,
                        "StationName": "Y8M-8XZ",
                        "StationType": "FleetCarrier",
                        "StarSystem": "Mel 22 Sector IN-S c4-4",
                        "Items": [
                            {
                                "id": 128672701,
                                "Name": "$metaalloys_name;",
                                "Name_Localised": "Meta-Alloys",
                                "Category": "$MARKET_category_industrial_materials;",
                                "Category_Localised": "Industrial materials",
                                "BuyPrice": 19545900,
                                "SellPrice": 0,
                                "MeanPrice": 0,
                                "StockBracket": 0,
                                "DemandBracket": 0,
                                "Stock": 0,
                                "Demand": 0,
                                "Consumer": False,
                                "Producer": False,
                                "Rare": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            journal_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-04-07T23:16:25Z",
                                "event": "Docked",
                                "StationName": "Y8M-8XZ",
                                "StarSystem": "Mel 22 Sector IN-S c4-4",
                                "MarketID": 3700490240,
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-04-07T23:19:51Z",
                                "event": "Market",
                                "StationName": "Y8M-8XZ",
                                "StarSystem": "Mel 22 Sector IN-S c4-4",
                                "MarketID": 3700490240,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            snapshot = MarketReader(market_path, journal_dir=root).snapshot(required=True)

            self.assertEqual(snapshot.station_name, "Y8M-8XZ")
            self.assertEqual(snapshot.star_system, "Mel 22 Sector IN-S c4-4")
            self.assertEqual(snapshot.market_id, 3700490240)
            self.assertEqual(len(snapshot.commodities), 1)
            self.assertEqual(snapshot.commodities[0].name, "metaalloys")
            self.assertEqual(snapshot.commodities[0].name_localised, "Meta-Alloys")
            self.assertTrue(snapshot.matches_last_docked_station())

    def test_snapshot_handles_missing_journal_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            market_path = root / "Market.json"
            market_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-04-07T23:19:51Z",
                        "event": "Market",
                        "StationName": "Any Station",
                        "Items": [],
                    }
                ),
                encoding="utf-8",
            )

            snapshot = MarketReader(market_path, journal_dir=root).snapshot(required=True)
            self.assertIsNone(snapshot.docked_context)
            self.assertIsNone(snapshot.matches_last_docked_station())

    def test_missing_market_file_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing_path = root / "Market.json"
            reader = MarketReader(missing_path, journal_dir=root)

            with self.assertRaises(FileNotFoundError):
                reader.snapshot(required=True)


if __name__ == "__main__":
    unittest.main()
