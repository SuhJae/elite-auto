from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ShipState:
    """High-level ship state derived from local Elite Dangerous files."""

    is_docked: bool
    is_mass_locked: bool
    is_supercruise: bool
    cargo_count: int
    gui_focus: int | None
    status_flags: int
    raw_status: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    source_timestamp: datetime | None = None
    cargo_path: Path | None = None
    cargo_timestamp: datetime | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "is_docked": self.is_docked,
            "is_mass_locked": self.is_mass_locked,
            "is_supercruise": self.is_supercruise,
            "cargo_count": self.cargo_count,
            "gui_focus": self.gui_focus,
            "status_flags": self.status_flags,
            "source_path": str(self.source_path) if self.source_path else None,
            "source_timestamp": self.source_timestamp.isoformat() if self.source_timestamp else None,
            "cargo_path": str(self.cargo_path) if self.cargo_path else None,
            "cargo_timestamp": self.cargo_timestamp.isoformat() if self.cargo_timestamp else None,
        }


@dataclass(slots=True)
class JournalEvent:
    """Typed wrapper around a single JSON journal event."""

    event_type: str
    timestamp: str | None
    payload: dict[str, Any]
    source_path: Path
    raw_line: str


@dataclass(slots=True)
class TemplateMatch:
    """Result of a template match within a capture frame."""

    confidence: float
    top_left: tuple[int, int]
    size: tuple[int, int]
    region: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class CommodityListing:
    """One commodity entry from Market.json."""

    commodity_id: int | None
    name: str
    name_localised: str | None
    category: str | None
    category_localised: str | None
    buy_price: int
    sell_price: int
    mean_price: int
    stock: int
    demand: int
    stock_bracket: int | None
    demand_bracket: int | None
    consumer: bool
    producer: bool
    rare: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "commodity_id": self.commodity_id,
            "name": self.name,
            "name_localised": self.name_localised,
            "category": self.category,
            "category_localised": self.category_localised,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "mean_price": self.mean_price,
            "stock": self.stock,
            "demand": self.demand,
            "stock_bracket": self.stock_bracket,
            "demand_bracket": self.demand_bracket,
            "consumer": self.consumer,
            "producer": self.producer,
            "rare": self.rare,
        }


@dataclass(slots=True)
class DockedStationContext:
    """Recent dock/market context inferred from the journal."""

    station_name: str | None
    star_system: str | None
    market_id: int | None
    docked_timestamp: str | None
    market_timestamp: str | None
    journal_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_name": self.station_name,
            "star_system": self.star_system,
            "market_id": self.market_id,
            "docked_timestamp": self.docked_timestamp,
            "market_timestamp": self.market_timestamp,
            "journal_path": str(self.journal_path),
        }


@dataclass(slots=True)
class MarketSnapshot:
    """Parsed commodity market snapshot with journal-derived context."""

    station_name: str | None
    station_type: str | None
    star_system: str | None
    market_id: int | None
    timestamp: str | None
    commodities: list[CommodityListing]
    source_path: Path
    source_timestamp: datetime | None
    docked_context: DockedStationContext | None = None

    def to_dict(self) -> dict[str, Any]:
        context = self.docked_context.to_dict() if self.docked_context else None
        return {
            "station_name": self.station_name,
            "station_type": self.station_type,
            "star_system": self.star_system,
            "market_id": self.market_id,
            "timestamp": self.timestamp,
            "commodity_count": len(self.commodities),
            "commodities": [commodity.to_dict() for commodity in self.commodities],
            "source_path": str(self.source_path),
            "source_timestamp": self.source_timestamp.isoformat() if self.source_timestamp else None,
            "docked_context": context,
            "matches_last_docked_station": self.matches_last_docked_station(),
        }

    def matches_last_docked_station(self) -> bool | None:
        if self.docked_context is None:
            return None
        station_match = self.station_name == self.docked_context.station_name
        system_match = self.star_system == self.docked_context.star_system
        market_match = self.market_id == self.docked_context.market_id
        return station_match and system_match and market_match
