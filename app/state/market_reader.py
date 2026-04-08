from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.domain.models import CommodityListing, DockedStationContext, MarketSnapshot


class MarketReader:
    """Read Market.json and correlate it with recent dock/market journal events."""

    def __init__(
        self,
        market_path: str | Path,
        journal_dir: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._market_path = Path(market_path)
        self._journal_dir = Path(journal_dir) if journal_dir is not None else None
        self._logger = logger or logging.getLogger(__name__)

    @property
    def path(self) -> Path:
        return self._market_path

    def read(self, required: bool = False) -> dict[str, Any]:
        if not self._market_path.exists():
            if required:
                raise FileNotFoundError(f"Market file not found: {self._market_path}")
            return {}

        try:
            with self._market_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Market file contains invalid JSON: {self._market_path}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Market file must contain a JSON object: {self._market_path}")
        return payload

    def station_name(self) -> str | None:
        payload = self.read(required=False)
        station_name = payload.get("StationName")
        return station_name if isinstance(station_name, str) else None

    def snapshot(self, required: bool = False) -> MarketSnapshot:
        payload = self.read(required=required)
        if not payload:
            raise FileNotFoundError(f"Market file not found: {self._market_path}")

        items = payload.get("Items", [])
        if items is None:
            items = []
        if not isinstance(items, list):
            raise ValueError(f"Market file Items must be a list: {self._market_path}")

        commodities = [self._parse_commodity(item) for item in items if isinstance(item, dict)]
        context = self.latest_docked_context()

        snapshot = MarketSnapshot(
            station_name=_as_str_or_none(payload.get("StationName")),
            station_type=_as_str_or_none(payload.get("StationType")),
            star_system=_as_str_or_none(payload.get("StarSystem")),
            market_id=_as_int_or_none(payload.get("MarketID")),
            timestamp=_as_str_or_none(payload.get("timestamp")),
            commodities=commodities,
            source_path=self._market_path,
            source_timestamp=self.modified_at(),
            docked_context=context,
        )
        self._logger.info(
            "Market snapshot loaded",
            extra={
                "station_name": snapshot.station_name,
                "star_system": snapshot.star_system,
                "commodity_count": len(snapshot.commodities),
                "matches_last_docked_station": snapshot.matches_last_docked_station(),
            },
        )
        return snapshot

    def latest_docked_context(self) -> DockedStationContext | None:
        latest_journal = self._find_latest_journal_file()
        if latest_journal is None:
            return None

        try:
            lines = latest_journal.read_text(encoding="utf-8").splitlines()
        except OSError:
            self._logger.warning("Could not read journal file for market context.", extra={"path": str(latest_journal)})
            return None

        latest_docked: dict[str, Any] | None = None
        latest_market: dict[str, Any] | None = None

        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            event_name = payload.get("event")
            if event_name == "Docked" and latest_docked is None:
                latest_docked = payload
            elif event_name == "Market" and latest_market is None:
                latest_market = payload
            if latest_docked is not None and latest_market is not None:
                break

        if latest_docked is None and latest_market is None:
            return None

        preferred = latest_market or latest_docked or {}
        return DockedStationContext(
            station_name=_as_str_or_none(preferred.get("StationName")) or _as_str_or_none((latest_docked or {}).get("StationName")),
            star_system=_as_str_or_none(preferred.get("StarSystem")) or _as_str_or_none((latest_docked or {}).get("StarSystem")),
            market_id=_as_int_or_none(preferred.get("MarketID")) or _as_int_or_none((latest_docked or {}).get("MarketID")),
            docked_timestamp=_as_str_or_none((latest_docked or {}).get("timestamp")),
            market_timestamp=_as_str_or_none((latest_market or {}).get("timestamp")),
            journal_path=latest_journal,
        )

    def modified_at(self) -> datetime | None:
        if not self._market_path.exists():
            return None
        return datetime.fromtimestamp(self._market_path.stat().st_mtime, tz=timezone.utc)

    def _find_latest_journal_file(self) -> Path | None:
        if self._journal_dir is None or not self._journal_dir.exists():
            return None

        candidates = sorted(
            self._journal_dir.glob("Journal*.log"),
            key=lambda path: (path.stat().st_mtime, path.name),
        )
        return candidates[-1] if candidates else None

    def _parse_commodity(self, item: dict[str, Any]) -> CommodityListing:
        return CommodityListing(
            commodity_id=_as_int_or_none(item.get("id")),
            name=_localize_symbol_name(_as_str_or_none(item.get("Name"))),
            name_localised=_as_str_or_none(item.get("Name_Localised")),
            category=_as_str_or_none(item.get("Category")),
            category_localised=_as_str_or_none(item.get("Category_Localised")),
            buy_price=_as_int(item.get("BuyPrice")),
            sell_price=_as_int(item.get("SellPrice")),
            mean_price=_as_int(item.get("MeanPrice")),
            stock=_as_int(item.get("Stock")),
            demand=_as_int(item.get("Demand")),
            stock_bracket=_as_int_or_none(item.get("StockBracket")),
            demand_bracket=_as_int_or_none(item.get("DemandBracket")),
            consumer=bool(item.get("Consumer", False)),
            producer=bool(item.get("Producer", False)),
            rare=bool(item.get("Rare", False)),
        )


def _as_int(value: Any) -> int:
    return int(value or 0)


def _as_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _as_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _localize_symbol_name(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.strip("$")
    if cleaned.endswith("_name;"):
        cleaned = cleaned[:-6]
    elif cleaned.endswith(";"):
        cleaned = cleaned[:-1]
    return cleaned
