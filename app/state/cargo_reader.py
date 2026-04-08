from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CargoReader:
    """Read cargo information from Cargo.json."""

    def __init__(self, cargo_path: str | Path) -> None:
        self._cargo_path = Path(cargo_path)

    @property
    def path(self) -> Path:
        return self._cargo_path

    def read(self, required: bool = False) -> dict[str, Any]:
        if not self._cargo_path.exists():
            if required:
                raise FileNotFoundError(f"Cargo file not found: {self._cargo_path}")
            return {}

        try:
            with self._cargo_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cargo file contains invalid JSON: {self._cargo_path}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Cargo file must contain a JSON object: {self._cargo_path}")
        return payload

    def cargo_count(self, required: bool = False) -> int:
        payload = self.read(required=required)
        inventory = payload.get("Inventory", [])
        if not isinstance(inventory, list):
            return 0
        return sum(int(item.get("Count", 0)) for item in inventory if isinstance(item, dict))

    def modified_at(self) -> datetime | None:
        if not self._cargo_path.exists():
            return None
        return datetime.fromtimestamp(self._cargo_path.stat().st_mtime, tz=timezone.utc)
