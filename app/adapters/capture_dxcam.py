from __future__ import annotations

import platform
from typing import Any

import dxcam

from app.domain.protocols import Capture, Region


class DxcamCapture(Capture):
    """Windows-first dxcam wrapper for future screen-based game inspection."""

    def __init__(self, output_index: int = 0, target_fps: int | None = None) -> None:
        self._output_index = output_index
        self._target_fps = target_fps
        self._camera: Any | None = None
        self._started = False

        if platform.system() != "Windows":
            raise OSError("DxcamCapture is only supported on Windows.")

    @property
    def output_index(self) -> int:
        return self._output_index

    def set_output_index(self, output_index: int) -> None:
        if output_index == self._output_index:
            return
        self.stop()
        self._camera = None
        self._started = False
        self._output_index = output_index

    def start(self) -> None:
        if self._started:
            return

        try:
            self._camera = dxcam.create(output_idx=self._output_index)
        except Exception as exc:  # pragma: no cover - depends on local DX environment
            raise RuntimeError("Failed to initialize dxcam capture.") from exc

        if self._camera is None:
            raise RuntimeError("dxcam did not return a capture device.")

        if self._target_fps:
            self._camera.start(target_fps=self._target_fps)

        self._started = True

    def stop(self) -> None:
        if self._camera is None:
            return
        if self._target_fps:
            self._camera.stop()
        self._started = False

    def grab(self, region: Region | None = None) -> Any:
        if self._camera is None:
            self.start()

        if self._camera is None:
            raise RuntimeError("Capture device is not available.")

        if self._target_fps:
            frame = self._camera.get_latest_frame()
        else:
            frame = self._camera.grab(region=region)

        if frame is None:
            raise RuntimeError("dxcam returned no frame. Check monitor/output availability.")
        return frame
