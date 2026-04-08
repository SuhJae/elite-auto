from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.domain.models import ShipState
from app.domain.protocols import StateReader
from app.state.cargo_reader import CargoReader

FLAG_DOCKED = 1 << 0
FLAG_SUPERCRUISE = 1 << 4
FLAG_FSD_MASS_LOCKED = 1 << 16


class StatusFileReader:
    """Low-level parser for Status.json."""

    def __init__(
        self,
        status_path: str | Path,
        retry_attempts: int = 5,
        retry_interval_seconds: float = 0.05,
    ) -> None:
        self._status_path = Path(status_path)
        self._retry_attempts = max(1, retry_attempts)
        self._retry_interval_seconds = max(0.0, retry_interval_seconds)

    @property
    def path(self) -> Path:
        return self._status_path

    def read(self) -> dict[str, Any]:
        last_error: ValueError | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                return self._read_payload_once()
            except ValueError as exc:
                last_error = exc
                if attempt >= self._retry_attempts:
                    break
                if self._retry_interval_seconds > 0:
                    time.sleep(self._retry_interval_seconds)

        assert last_error is not None
        raise last_error

    def _read_payload_once(self) -> dict[str, Any]:
        if not self._status_path.exists():
            raise FileNotFoundError(f"Status file not found: {self._status_path}")

        try:
            with self._status_path.open("r", encoding="utf-8") as handle:
                raw_text = handle.read()
        except json.JSONDecodeError as exc:
            raise ValueError(f"Status file contains invalid JSON: {self._status_path}") from exc

        if not raw_text.strip():
            raise ValueError(f"Status file contains invalid JSON: {self._status_path}")

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Status file contains invalid JSON: {self._status_path}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Status file must contain a JSON object: {self._status_path}")
        return payload

    def modified_at(self) -> datetime:
        return datetime.fromtimestamp(self._status_path.stat().st_mtime, tz=timezone.utc)


class EliteStateReader(StateReader):
    """File-backed ship state composed from Status.json and Cargo.json."""

    def __init__(
        self,
        status_reader: StatusFileReader,
        cargo_reader: CargoReader | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._status_reader = status_reader
        self._cargo_reader = cargo_reader
        self._logger = logger or logging.getLogger(__name__)

    def snapshot(self) -> ShipState:
        payload = self._status_reader.read()
        flags = int(payload.get("Flags", 0))
        cargo_count = 0
        cargo_timestamp = None
        cargo_path = None

        if self._cargo_reader is not None:
            cargo_count = self._cargo_reader.cargo_count(required=False)
            cargo_timestamp = self._cargo_reader.modified_at()
            cargo_path = self._cargo_reader.path

        state = ShipState(
            is_docked=bool(flags & FLAG_DOCKED),
            is_mass_locked=bool(flags & FLAG_FSD_MASS_LOCKED),
            is_supercruise=bool(flags & FLAG_SUPERCRUISE),
            cargo_count=cargo_count,
            gui_focus=_as_int_or_none(payload.get("GuiFocus")),
            status_flags=flags,
            raw_status=payload,
            source_path=self._status_reader.path,
            source_timestamp=self._status_reader.modified_at(),
            cargo_path=cargo_path,
            cargo_timestamp=cargo_timestamp,
        )
        self._logger.debug("State snapshot loaded", extra={"state": state.to_debug_dict()})
        return state

    def is_docked(self) -> bool:
        return self.snapshot().is_docked

    def is_mass_locked(self) -> bool:
        return self.snapshot().is_mass_locked

    def is_supercruise(self) -> bool:
        return self.snapshot().is_supercruise

    def cargo_count(self) -> int:
        return self.snapshot().cargo_count

    def gui_focus(self) -> int | None:
        return self.snapshot().gui_focus


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
