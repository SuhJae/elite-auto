from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.domain.protocols import Capture, EventStream, Input, ShipControl, StateReader, VisionSystem

if TYPE_CHECKING:
    from app.config import AppConfig


@dataclass(slots=True)
class Context:
    """Shared dependencies passed into actions and routines."""

    config: "AppConfig"
    logger: logging.Logger
    debug_snapshot_dir: Path
    state_reader: StateReader
    event_stream: EventStream
    input_adapter: Input | None = None
    ship_control: ShipControl | None = None
    capture: Capture | None = None
    vision: VisionSystem | None = None
