from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.domain.models import JournalEvent, MarketSnapshot, ShipState, TemplateMatch

Region = tuple[int, int, int, int]


@runtime_checkable
class Capture(Protocol):
    """Capture frames from the screen or a configured region."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def grab(self, region: Region | None = None) -> Any:
        ...


@runtime_checkable
class Input(Protocol):
    """Low-level input adapter for keyboard-style game controls."""

    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        ...

    def key_down(self, key: str) -> None:
        ...

    def key_up(self, key: str) -> None:
        ...

    def hold(self, key: str, seconds: float) -> None:
        ...


@runtime_checkable
class ShipControl(Protocol):
    """Game-aware controls exposed to actions and routines."""

    def set_throttle_percent(self, percent: int) -> None:
        ...

    def boost(self) -> None:
        ...

    def open_left_panel(self) -> None:
        ...

    def cycle_previous_panel(self) -> None:
        ...

    def cycle_next_panel(self) -> None:
        ...

    def ui_select(self, direction: str = "select") -> None:
        ...

    def charge_fsd(self, target: str = "supercruise") -> None:
        ...


@runtime_checkable
class StateReader(Protocol):
    """High-level ship state backed by local game files."""

    def snapshot(self) -> ShipState:
        ...

    def is_docked(self) -> bool:
        ...

    def is_mass_locked(self) -> bool:
        ...

    def is_supercruise(self) -> bool:
        ...

    def cargo_count(self) -> int:
        ...

    def gui_focus(self) -> int | None:
        ...


@runtime_checkable
class EventStream(Protocol):
    """Read newly appended structured events from the journal."""

    def poll_events(self, limit: int | None = None) -> Sequence[JournalEvent]:
        ...


@runtime_checkable
class VisionSystem(Protocol):
    """Computer vision utilities kept behind an adapter boundary."""

    def match_template(
        self,
        image: Any,
        template: Any,
        region: Region | None = None,
        threshold: float = 0.9,
    ) -> TemplateMatch | None:
        ...

    def save_debug_snapshot(self, name: str, image: Any) -> Path:
        ...


@runtime_checkable
class MarketDataSource(Protocol):
    """Read a parsed market snapshot from local files."""

    def snapshot(self, required: bool = False) -> MarketSnapshot:
        ...
