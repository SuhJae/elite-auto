from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from app.state.journal_tailer import JournalTailer


class TestJournalTailer(unittest.TestCase):
    def test_find_latest_journal_file_prefers_newest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            older = root / "Journal.2026-04-07T100000.01.log"
            newer = root / "Journal.2026-04-07T110000.01.log"
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            now = time.time()
            os.utime(older, (now - 10, now - 10))
            os.utime(newer, (now, now))

            tailer = JournalTailer(root)
            self.assertEqual(tailer.find_latest_journal_file(), newer)

    def test_start_at_end_does_not_replay_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            journal = root / "Journal.2026-04-07T110000.01.log"
            journal.write_text('{"event":"Existing"}\n', encoding="utf-8")

            tailer = JournalTailer(root, start_at_end=True)
            initial = tailer.poll_events()
            self.assertEqual(initial, [])

            with journal.open("a", encoding="utf-8") as handle:
                handle.write('{"timestamp":"2026-04-07T11:01:00Z","event":"NewEntry"}\n')

            events = tailer.poll_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "NewEntry")

    def test_new_lines_are_parsed_and_malformed_lines_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            journal = root / "Journal.2026-04-07T120000.01.log"
            journal.write_text("", encoding="utf-8")

            tailer = JournalTailer(root, start_at_end=False)
            tailer.open_latest(start_at_end=False)

            with journal.open("a", encoding="utf-8") as handle:
                handle.write('{"timestamp":"2026-04-07T12:00:01Z","event":"Docked"}\n')
                handle.write('{"bad_json"\n')
                handle.write('{"timestamp":"2026-04-07T12:00:02Z","event":"Undocked"}\n')

            events = tailer.poll_events()
            self.assertEqual([event.event_type for event in events], ["Docked", "Undocked"])


if __name__ == "__main__":
    unittest.main()
