# Focus the Elite Dangerous game window before this countdown ends.
from __future__ import annotations

import sys
import time

from app.adapters.input_pydirect import PydirectInputAdapter


def main() -> int:
    try:
        input_adapter = PydirectInputAdapter()

        print("Warning: this will send W and Tab to the currently focused window.")
        for seconds_remaining in range(5, 0, -1):
            print(f"Starting in {seconds_remaining}...")
            time.sleep(1)

        print("Holding W for 3 seconds...")
        input_adapter.hold("w", 3.0)

        time.sleep(0.5)

        print("Pressing Tab once...")
        input_adapter.press("tab")

        print("done")
        return 0
    except Exception as exc:
        print(f"Input test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
