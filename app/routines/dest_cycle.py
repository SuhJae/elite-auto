from __future__ import annotations

from app.domain.context import Context
from app.domain.result import Result


class DestinationCyclePlaceholder:
    """Future destination-side routine placeholder."""

    name = "destination_cycle_placeholder"

    def run(self, context: Context) -> Result:
        return Result.fail(
            "Destination routine is not implemented in this scaffold.",
            debug={"todo": "Compose docking, approach, and sell/buy action modules later."},
        )
