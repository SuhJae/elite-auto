"""Minimal autoclicker that right-clicks every 400 seconds."""

from __future__ import annotations

import time

import pydirectinput


INTERVAL_SECONDS = 400
START_DELAY_SECONDS = 3


def main() -> None:
    pydirectinput.PAUSE = 0
    print(
        f"Starting in {START_DELAY_SECONDS} seconds, then right-clicking every "
        f"{INTERVAL_SECONDS} seconds. Press Ctrl+C to stop."
    )

    try:
        time.sleep(START_DELAY_SECONDS)
        while True:
            pydirectinput.click(button="right")
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
