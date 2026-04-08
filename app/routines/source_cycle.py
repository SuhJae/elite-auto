from __future__ import annotations

from app.actions.fsd import WaitForMassLockClear
from app.actions.launch import WaitUntilUndocked
from app.actions.navigation import WaitForSupercruiseEntry
from app.domain.context import Context
from app.domain.result import Result


class SampleDepartureRoutine:
    """Example routine that composes a few small departure checks."""

    name = "sample_departure"

    def __init__(self, timeout_seconds: float, poll_interval_seconds: float) -> None:
        self._steps = [
            WaitUntilUndocked(timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds),
            WaitForMassLockClear(timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds),
            WaitForSupercruiseEntry(timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds),
        ]

    def run(self, context: Context) -> Result:
        for step in self._steps:
            context.logger.info("Running routine step", extra={"step": step.name})
            result = step.run(context)
            if not result.success:
                return Result.fail(
                    f"Routine failed at step '{step.name}': {result.reason}",
                    debug=result.debug,
                )
        return Result.ok("Sample departure routine completed.")
