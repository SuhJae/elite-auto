from __future__ import annotations

import time

from app.domain.context import Context
from app.domain.result import Result


class WaitUntilUndocked:
    """Poll state until the ship is no longer docked."""

    name = "wait_until_undocked"

    def __init__(self, timeout_seconds: float, poll_interval_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def run(self, context: Context) -> Result:
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            state = context.state_reader.snapshot()
            if not state.is_docked:
                context.logger.info("Undocked state detected", extra={"state": state.to_debug_dict()})
                return Result.ok("Ship is undocked.", debug=state.to_debug_dict())
            time.sleep(self._poll_interval_seconds)

        return Result.fail(
            "Timed out waiting for ship to undock.",
            debug={"timeout_seconds": self._timeout_seconds},
        )
