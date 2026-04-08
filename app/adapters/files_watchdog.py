from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from queue import Queue

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


class _QueueingEventHandler(FileSystemEventHandler):
    def __init__(self, sink: Callable[[Path], None]) -> None:
        self._sink = sink

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._sink(Path(event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._sink(Path(event.src_path))


class WatchdogFileWatcher:
    """Watch one or more directories and queue changed file paths."""

    def __init__(self, paths: Iterable[Path], recursive: bool = False) -> None:
        self._paths = [Path(path) for path in paths]
        self._recursive = recursive
        self._observer = Observer()
        self._events: Queue[Path] = Queue()
        self._handler = _QueueingEventHandler(self._events.put)
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        for path in self._paths:
            if not path.exists():
                raise FileNotFoundError(f"Watch path does not exist: {path}")
            self._observer.schedule(self._handler, str(path), recursive=self._recursive)
        self._observer.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._observer.stop()
        self._observer.join(timeout=5)
        self._started = False

    def drain_events(self) -> list[Path]:
        events: list[Path] = []
        while not self._events.empty():
            events.append(self._events.get_nowait())
        return events
