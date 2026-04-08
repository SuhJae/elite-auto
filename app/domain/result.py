from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Result:
    """Standard action and routine result payload."""

    success: bool
    reason: str
    debug: dict[str, Any] | None = None

    @classmethod
    def ok(cls, reason: str, debug: dict[str, Any] | None = None) -> "Result":
        return cls(success=True, reason=reason, debug=debug)

    @classmethod
    def fail(cls, reason: str, debug: dict[str, Any] | None = None) -> "Result":
        return cls(success=False, reason=reason, debug=debug)

    def format_debug(self) -> str:
        if not self.debug:
            return ""
        return json.dumps(self.debug, indent=2, ensure_ascii=False)
