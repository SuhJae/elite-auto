from __future__ import annotations

import json
import logging
from pathlib import Path

from app.domain.models import JournalEvent
from app.domain.protocols import EventStream


class JournalTailer(EventStream):
    """Tail the newest Elite Dangerous journal file and emit appended events."""

    def __init__(
        self,
        journal_dir: str | Path,
        logger: logging.Logger | None = None,
        start_at_end: bool = True,
    ) -> None:
        self._journal_dir = Path(journal_dir)
        self._logger = logger or logging.getLogger(__name__)
        self._start_at_end = start_at_end
        self._current_path: Path | None = None
        self._offset = 0
        self._initialized = False

    @property
    def current_path(self) -> Path | None:
        return self._current_path

    def find_latest_journal_file(self) -> Path:
        if not self._journal_dir.exists():
            raise FileNotFoundError(f"Journal directory not found: {self._journal_dir}")

        candidates = sorted(
            self._journal_dir.glob("Journal*.log"),
            key=lambda path: (path.stat().st_mtime, path.name),
        )
        if not candidates:
            raise FileNotFoundError(f"No journal files found in: {self._journal_dir}")
        return candidates[-1]

    def open_latest(self, start_at_end: bool | None = None) -> Path:
        latest = self.find_latest_journal_file()
        should_seek_end = self._start_at_end if start_at_end is None else start_at_end
        self._current_path = latest
        self._offset = latest.stat().st_size if should_seek_end else 0
        self._initialized = True
        return latest

    def poll_events(self, limit: int | None = None) -> list[JournalEvent]:
        if not self._initialized:
            self.open_latest(start_at_end=self._start_at_end)
        else:
            latest = self.find_latest_journal_file()
            if latest != self._current_path:
                self._current_path = latest
                self._offset = 0

        if self._current_path is None:
            raise RuntimeError("Journal tailer is not initialized.")

        events: list[JournalEvent] = []
        with self._current_path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offset)

            while True:
                line = handle.readline()
                if not line:
                    break

                self._offset = handle.tell()
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    self._logger.warning(
                        "Skipping malformed journal line",
                        extra={"path": str(self._current_path), "line": stripped},
                    )
                    continue

                if not isinstance(payload, dict):
                    self._logger.warning(
                        "Skipping non-object journal event",
                        extra={"path": str(self._current_path), "line": stripped},
                    )
                    continue

                event = JournalEvent(
                    event_type=str(payload.get("event", "Unknown")),
                    timestamp=payload.get("timestamp"),
                    payload=payload,
                    source_path=self._current_path,
                    raw_line=stripped,
                )
                events.append(event)
                if limit is not None and len(events) >= limit:
                    break

        return events
