from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any


DEFAULT_ELITE_DIR = Path.home() / "Saved Games" / "Frontier Developments" / "Elite Dangerous"
DEFAULT_BINDINGS_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Frontier Developments" / "Elite Dangerous" / "Options" / "Bindings"
STANDARD_LOG_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


@dataclass(slots=True)
class PathConfig:
    """Filesystem locations used by the application."""

    status_file: Path = DEFAULT_ELITE_DIR / "Status.json"
    journal_dir: Path = DEFAULT_ELITE_DIR
    cargo_file: Path = DEFAULT_ELITE_DIR / "Cargo.json"
    market_file: Path = DEFAULT_ELITE_DIR / "Market.json"
    bindings_dir: Path = DEFAULT_BINDINGS_DIR
    start_preset_file: Path = DEFAULT_BINDINGS_DIR / "StartPreset.4.start"
    logs_dir: Path = Path("logs")
    debug_snapshots_dir: Path = Path("debug_snapshots")


@dataclass(slots=True)
class RuntimeConfig:
    """Polling and timeout defaults for actions and routines."""

    action_timeout_seconds: float = 30.0
    poll_interval_seconds: float = 0.5


@dataclass(slots=True)
class ControlConfig:
    """Keyboard bindings used by the ship control adapter.

    TODO: Replace these defaults by parsing the user's `.binds` file.
    """

    throttle_zero: str = "x"
    throttle_fifty: str = "c"
    throttle_seventy_five: str = "v"
    throttle_full: str = "w"
    throttle_reverse_full: str = "9"
    boost: str = "tab"
    open_left_panel: str = "1"
    cycle_previous_panel: str = "q"
    cycle_next_panel: str = "e"
    ui_up: str = "up"
    ui_down: str = "down"
    ui_left: str = "left"
    ui_right: str = "right"
    ui_select: str = "space"
    ui_back: str = "backspace"
    charge_fsd: str = "j"
    thrust_up: str = "r"
    pitch_up: str = "y"
    pitch_down: str = "h"
    yaw_left: str = "a"
    yaw_right: str = "d"


@dataclass(slots=True)
class AppConfig:
    """Application configuration loaded from defaults or JSON."""

    paths: PathConfig = field(default_factory=PathConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    controls: ControlConfig = field(default_factory=ControlConfig)

    @classmethod
    def default(cls) -> "AppConfig":
        return cls()

    @classmethod
    def from_json(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        if not isinstance(raw, dict):
            raise ValueError(f"Config file must contain a JSON object: {config_path}")

        base_dir = config_path.parent
        default_paths = PathConfig()
        default_runtime = RuntimeConfig()
        default_controls = ControlConfig()
        paths_raw = raw.get("paths", {})
        runtime_raw = raw.get("runtime", {})
        controls_raw = raw.get("controls", {})

        return cls(
            paths=PathConfig(
                status_file=_resolve_path(paths_raw.get("status_file"), default_paths.status_file, base_dir),
                journal_dir=_resolve_path(paths_raw.get("journal_dir"), default_paths.journal_dir, base_dir),
                cargo_file=_resolve_path(paths_raw.get("cargo_file"), default_paths.cargo_file, base_dir),
                market_file=_resolve_path(paths_raw.get("market_file"), default_paths.market_file, base_dir),
                bindings_dir=_resolve_path(paths_raw.get("bindings_dir"), default_paths.bindings_dir, base_dir),
                start_preset_file=_resolve_path(
                    paths_raw.get("start_preset_file"),
                    default_paths.start_preset_file,
                    base_dir,
                ),
                logs_dir=_resolve_path(paths_raw.get("logs_dir"), default_paths.logs_dir, base_dir),
                debug_snapshots_dir=_resolve_path(
                    paths_raw.get("debug_snapshots_dir"),
                    default_paths.debug_snapshots_dir,
                    base_dir,
                ),
            ),
            runtime=RuntimeConfig(
                action_timeout_seconds=float(runtime_raw.get("action_timeout_seconds", default_runtime.action_timeout_seconds)),
                poll_interval_seconds=float(runtime_raw.get("poll_interval_seconds", default_runtime.poll_interval_seconds)),
            ),
            controls=ControlConfig(
                throttle_zero=str(controls_raw.get("throttle_zero", default_controls.throttle_zero)),
                throttle_fifty=str(controls_raw.get("throttle_fifty", default_controls.throttle_fifty)),
                throttle_seventy_five=str(controls_raw.get("throttle_seventy_five", default_controls.throttle_seventy_five)),
                throttle_full=str(controls_raw.get("throttle_full", default_controls.throttle_full)),
                throttle_reverse_full=str(controls_raw.get("throttle_reverse_full", default_controls.throttle_reverse_full)),
                boost=str(controls_raw.get("boost", default_controls.boost)),
                open_left_panel=str(controls_raw.get("open_left_panel", default_controls.open_left_panel)),
                cycle_previous_panel=str(controls_raw.get("cycle_previous_panel", default_controls.cycle_previous_panel)),
                cycle_next_panel=str(controls_raw.get("cycle_next_panel", default_controls.cycle_next_panel)),
                ui_up=str(controls_raw.get("ui_up", default_controls.ui_up)),
                ui_down=str(controls_raw.get("ui_down", default_controls.ui_down)),
                ui_left=str(controls_raw.get("ui_left", default_controls.ui_left)),
                ui_right=str(controls_raw.get("ui_right", default_controls.ui_right)),
                ui_select=str(controls_raw.get("ui_select", default_controls.ui_select)),
                ui_back=str(controls_raw.get("ui_back", default_controls.ui_back)),
                charge_fsd=str(controls_raw.get("charge_fsd", default_controls.charge_fsd)),
                thrust_up=str(controls_raw.get("thrust_up", default_controls.thrust_up)),
                pitch_up=str(controls_raw.get("pitch_up", default_controls.pitch_up)),
                pitch_down=str(controls_raw.get("pitch_down", default_controls.pitch_down)),
                yaw_left=str(controls_raw.get("yaw_left", default_controls.yaw_left)),
                yaw_right=str(controls_raw.get("yaw_right", default_controls.yaw_right)),
            ),
        )

    def ensure_runtime_dirs(self) -> None:
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.debug_snapshots_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["paths"] = {name: str(value) for name, value in payload["paths"].items()}
        return payload


class StructuredFormatter(logging.Formatter):
    """Emit JSON-style log lines without extra dependencies."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in STANDARD_LOG_KEYS and not key.startswith("_")
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Emit compact human-readable console logs."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in STANDARD_LOG_KEYS and not key.startswith("_")
        }
        extras_text = ""
        if extras:
            formatted_extras = ", ".join(f"{key}={value}" for key, value in sorted(extras.items()))
            extras_text = f" | {formatted_extras}"
        message = f"{timestamp} | {record.levelname:<7} | {record.getMessage()}{extras_text}"
        if record.exc_info:
            return f"{message}\n{self.formatException(record.exc_info)}"
        return message


def configure_logging(config: AppConfig, logger_name: str = "elite_auto") -> logging.Logger:
    """Create console and file logging with a lightweight structured formatter."""

    config.ensure_runtime_dirs()
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    console_formatter = ConsoleFormatter()
    file_formatter = StructuredFormatter()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    log_file = config.paths.logs_dir / "elite_auto.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def _resolve_path(raw_value: Any, default: Path, base_dir: Path) -> Path:
    value = default if raw_value is None else Path(str(raw_value)).expanduser()
    return value if value.is_absolute() else (base_dir / value).resolve()
