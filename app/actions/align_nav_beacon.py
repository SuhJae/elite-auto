from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.actions.navigation_ocr import _find_window_client_region, _resolve_output_region
from app.actions.starport_buy import build_standalone_context
from app.adapters.capture_dxcam import DxcamCapture
from app.config import AppConfig
from app.domain.context import Context
from app.domain.result import Result
from app.domain.protocols import Region


# Article reference:
# https://networkgeekstuff.com/projects/autopilot-for-elite-dangerous-using-opencv-and-thoughts-on-cv-enabled-bots-in-visual-to-keyboard-loop/
#
# This implementation follows the article's general idea for "the thing in the center where
# you want to go": detect a known HUD element from a template, estimate its on-screen center,
# and use that offset to steer pitch/yaw. It intentionally stays scoped to nav beacon alignment.

STANDALONE_CONFIG_PATH: str | None = None
STANDALONE_TEMPLATE_PATH: str | None = None
STANDALONE_START_DELAY_SECONDS = 3.0

WINDOW_TITLE = "Elite Dangerous"


@dataclass(slots=True)
class NavBeaconAlignConfig:
    center_x: int = 960
    center_y: int = 540
    search_width: int = 900
    search_height: int = 700
    template_path: str | None = None
    alignment_tolerance_px: float = 18.0
    axis_alignment_tolerance_px: float = 12.0
    confirmation_reads: int = 3
    settle_seconds_after_input: float = 0.8
    idle_read_backoff_seconds: float = 0.15
    timeout_seconds: float | None = 20.0
    capture_retry_attempts: int = 3
    capture_retry_interval_seconds: float = 0.2
    max_consecutive_missing_reads: int = 10
    orb_features: int = 600
    lowe_ratio: float = 0.75
    min_match_count: int = 8
    min_inlier_count: int = 6
    min_inlier_ratio: float = 0.45
    yaw_seconds_per_pixel: float = 0.015
    pitch_seconds_per_pixel: float = 0.012
    min_pulse_seconds: float = 0.04
    max_pulse_seconds: float = 0.35
    pulse_bias_px: float = 6.0
    min_width: int = 1920
    min_height: int = 1080

    def search_region(self) -> Region:
        half_width = self.search_width // 2
        half_height = self.search_height // 2
        return (
            max(0, self.center_x - half_width),
            max(0, self.center_y - half_height),
            self.search_width,
            self.search_height,
        )


@dataclass(slots=True)
class NavBeaconDetection:
    detected: bool
    center_x: float | None = None
    center_y: float | None = None
    dx: float | None = None
    dy: float | None = None
    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    confidence: float = 0.0
    status: str = "missing"

    @property
    def distance(self) -> float:
        if self.dx is None or self.dy is None:
            return math.inf
        return math.hypot(self.dx, self.dy)

    def is_aligned(self, config: NavBeaconAlignConfig) -> bool:
        return (
            self.detected
            and self.dx is not None
            and self.dy is not None
            and abs(self.dx) < config.axis_alignment_tolerance_px
            and abs(self.dy) < config.axis_alignment_tolerance_px
            and self.distance < config.alignment_tolerance_px
        )

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "status": self.status,
            "center_x": None if self.center_x is None else round(self.center_x, 3),
            "center_y": None if self.center_y is None else round(self.center_y, 3),
            "dx": None if self.dx is None else round(self.dx, 3),
            "dy": None if self.dy is None else round(self.dy, 3),
            "distance": round(self.distance, 3) if math.isfinite(self.distance) else None,
            "match_count": self.match_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass(slots=True)
class PulseCommand:
    axis: str
    key_name: str
    pulse_seconds: float
    axis_error_px: float

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "key_name": self.key_name,
            "pulse_seconds": round(self.pulse_seconds, 4),
            "axis_error_px": round(self.axis_error_px, 3),
        }


@dataclass(slots=True)
class AlignToNavBeacon:
    config: NavBeaconAlignConfig = field(default_factory=NavBeaconAlignConfig)

    name = "align_to_nav_beacon"

    def run(self, context: Context) -> Result:
        capture = context.capture
        input_adapter = context.input_adapter
        if capture is None or input_adapter is None:
            return Result.fail("Nav beacon alignment requires both capture and input control.")

        initial_state = context.state_reader.snapshot()
        if initial_state.is_docked:
            return Result.fail(
                "Cannot align to the nav beacon while docked.",
                debug={"state": initial_state.to_debug_dict()},
            )

        try:
            output_index, capture_region = self._resolve_capture_region(context)
        except RuntimeError as exc:
            return Result.fail(str(exc))

        try:
            template = _load_template_image(self.config)
        except RuntimeError as exc:
            return Result.fail(str(exc))

        context.logger.info(
            "Starting nav beacon alignment",
            extra={
                "output_index": output_index,
                "capture_region": capture_region,
                "search_region": self.config.search_region(),
                "template_path": str(Path(self.config.template_path).expanduser()) if self.config.template_path else None,
                "align_config": {
                    "center_x": self.config.center_x,
                    "center_y": self.config.center_y,
                    "search_width": self.config.search_width,
                    "search_height": self.config.search_height,
                    "alignment_tolerance_px": self.config.alignment_tolerance_px,
                    "axis_alignment_tolerance_px": self.config.axis_alignment_tolerance_px,
                    "confirmation_reads": self.config.confirmation_reads,
                    "orb_features": self.config.orb_features,
                    "lowe_ratio": self.config.lowe_ratio,
                    "min_match_count": self.config.min_match_count,
                    "min_inlier_count": self.config.min_inlier_count,
                    "min_inlier_ratio": self.config.min_inlier_ratio,
                },
            },
        )

        deadline = None if self.config.timeout_seconds is None else (time.monotonic() + self.config.timeout_seconds)
        confirmation_count = 0
        consecutive_missing = 0
        last_debug: dict[str, Any] = {
            "output_index": output_index,
            "capture_region": capture_region,
            "search_region": self.config.search_region(),
            "initial_state": initial_state.to_debug_dict(),
        }

        while deadline is None or time.monotonic() < deadline:
            try:
                frame = self._capture_frame(capture, capture_region)
            except RuntimeError as exc:
                last_debug["capture_error"] = str(exc)
                return Result.fail("Capture failed while aligning to the nav beacon.", debug=last_debug)

            detection = detect_nav_beacon(frame, template, self.config)
            last_debug["detection"] = detection.to_debug_dict()

            context.logger.info(
                "Nav beacon read",
                extra=detection.to_debug_dict(),
            )

            if not detection.detected:
                confirmation_count = 0
                consecutive_missing += 1
                if consecutive_missing >= self.config.max_consecutive_missing_reads:
                    snapshot_path = _save_debug_snapshot(context, "nav_beacon_missing", frame)
                    last_debug["debug_snapshot"] = str(snapshot_path)
                    return Result.fail("Nav beacon could not be detected reliably.", debug=last_debug)
                time.sleep(self.config.idle_read_backoff_seconds)
                continue

            consecutive_missing = 0

            if detection.is_aligned(self.config):
                confirmation_count += 1
                context.logger.info(
                    "Nav beacon alignment confirmation",
                    extra={
                        "confirmation_count": confirmation_count,
                        "required_confirmation_reads": self.config.confirmation_reads,
                        "dx": round(detection.dx or 0.0, 3),
                        "dy": round(detection.dy or 0.0, 3),
                    },
                )
                if confirmation_count >= self.config.confirmation_reads:
                    return Result.ok("Ship aligned to the nav beacon.", debug=last_debug)
                time.sleep(self.config.idle_read_backoff_seconds)
                continue

            confirmation_count = 0
            command = _select_alignment_command(detection, self.config)
            key = getattr(context.config.controls, command.key_name)
            last_debug["command"] = command.to_debug_dict()

            context.logger.info(
                "Nav beacon alignment command",
                extra={
                    "dx": round(detection.dx or 0.0, 3),
                    "dy": round(detection.dy or 0.0, 3),
                    "distance": round(detection.distance, 3),
                    "command": command.to_debug_dict(),
                },
            )

            input_adapter.hold(key, command.pulse_seconds)
            time.sleep(self.config.settle_seconds_after_input)

        timeout_frame = self._capture_frame(capture, capture_region)
        snapshot_path = _save_debug_snapshot(context, "nav_beacon_timeout", timeout_frame)
        last_debug["debug_snapshot"] = str(snapshot_path)
        return Result.fail("Timed out while aligning to the nav beacon.", debug=last_debug)

    def _resolve_capture_region(self, context: Context) -> tuple[int, Region]:
        capture = context.capture
        if capture is None:
            raise RuntimeError("Capture is not available.")

        window_capture_region = _find_window_client_region(WINDOW_TITLE)
        if window_capture_region is None:
            raise RuntimeError(f"Could not locate the {WINDOW_TITLE!r} window.")

        output_index, output_relative_region = _resolve_output_region(window_capture_region)
        if output_index is None or output_relative_region is None:
            raise RuntimeError("Could not map the Elite window onto a dxcam output.")

        if output_relative_region[2] < self.config.min_width or output_relative_region[3] < self.config.min_height:
            raise RuntimeError(
                "Nav beacon alignment currently supports only the calibrated 1920x1080 HUD layout."
            )

        if isinstance(capture, DxcamCapture):
            capture.set_output_index(output_index)

        return output_index, output_relative_region

    def _capture_frame(self, capture: Any, capture_region: Region) -> Any:
        last_error: RuntimeError | None = None
        attempts = max(1, self.config.capture_retry_attempts)
        for attempt in range(1, attempts + 1):
            try:
                frame = capture.grab(region=capture_region)
            except RuntimeError as exc:
                last_error = exc
                frame = None
            if frame is not None:
                return frame
            if attempt < attempts:
                time.sleep(self.config.capture_retry_interval_seconds)

        if last_error is not None:
            raise RuntimeError(str(last_error))
        raise RuntimeError("Capture returned no frame.")


def detect_nav_beacon(image: Any, template: np.ndarray, config: NavBeaconAlignConfig) -> NavBeaconDetection:
    if image is None:
        return NavBeaconDetection(detected=False, status="missing")

    search_region = _clip_region(config.search_region(), image.shape[1], image.shape[0])
    roi = _extract_region(image, search_region)
    if roi.size == 0:
        return NavBeaconDetection(detected=False, status="missing")

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template

    orb = cv2.ORB_create(nfeatures=config.orb_features)
    template_keypoints, template_descriptors = orb.detectAndCompute(gray_template, None)
    roi_keypoints, roi_descriptors = orb.detectAndCompute(gray_roi, None)
    if template_descriptors is None or roi_descriptors is None:
        return NavBeaconDetection(detected=False, status="missing")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(template_descriptors, roi_descriptors, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < (config.lowe_ratio * second.distance):
            good_matches.append(first)

    if len(good_matches) < config.min_match_count:
        return NavBeaconDetection(
            detected=False,
            status="missing",
            match_count=len(good_matches),
        )

    src_points = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([roi_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        return NavBeaconDetection(
            detected=False,
            status="missing",
            match_count=len(good_matches),
        )

    inlier_count = int(mask.sum())
    inlier_ratio = inlier_count / max(len(good_matches), 1)
    if inlier_count < config.min_inlier_count or inlier_ratio < config.min_inlier_ratio:
        return NavBeaconDetection(
            detected=False,
            status="missing",
            match_count=len(good_matches),
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
        )

    template_height, template_width = gray_template.shape[:2]
    template_corners = np.float32(
        [
            [0.0, 0.0],
            [template_width - 1.0, 0.0],
            [template_width - 1.0, template_height - 1.0],
            [0.0, template_height - 1.0],
        ]
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(template_corners, homography).reshape(-1, 2)
    center_local = projected.mean(axis=0)
    center_x = search_region[0] + float(center_local[0])
    center_y = search_region[1] + float(center_local[1])
    dx = center_x - config.center_x
    dy = center_y - config.center_y
    confidence = inlier_ratio * (inlier_count / max(config.min_inlier_count, 1))

    return NavBeaconDetection(
        detected=True,
        center_x=center_x,
        center_y=center_y,
        dx=dx,
        dy=dy,
        match_count=len(good_matches),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        confidence=confidence,
        status="detected",
    )


def _select_alignment_command(detection: NavBeaconDetection, config: NavBeaconAlignConfig) -> PulseCommand:
    assert detection.dx is not None
    assert detection.dy is not None

    x_error = abs(detection.dx)
    y_error = abs(detection.dy)
    if x_error >= y_error:
        axis = "yaw"
        key_name = "yaw_left" if detection.dx < 0 else "yaw_right"
        axis_error = x_error
        seconds_per_pixel = config.yaw_seconds_per_pixel
    else:
        axis = "pitch"
        key_name = "pitch_up" if detection.dy < 0 else "pitch_down"
        axis_error = y_error
        seconds_per_pixel = config.pitch_seconds_per_pixel

    effective_error = max(0.0, axis_error - config.pulse_bias_px)
    pulse_seconds = effective_error * seconds_per_pixel
    pulse_seconds = min(config.max_pulse_seconds, max(config.min_pulse_seconds, pulse_seconds))
    return PulseCommand(
        axis=axis,
        key_name=key_name,
        pulse_seconds=pulse_seconds,
        axis_error_px=axis_error,
    )


def _load_template_image(config: NavBeaconAlignConfig) -> np.ndarray:
    if not config.template_path:
        raise RuntimeError(
            "Nav beacon alignment requires 'template_path' to point to a cropped nav beacon template image."
        )
    path = Path(config.template_path).expanduser().resolve()
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        raise RuntimeError(f"Could not load nav beacon template: {path}")
    return template


def _extract_region(image: Any, region: Region) -> Any:
    x, y, width, height = region
    return image[y : y + height, x : x + width]


def _clip_region(region: Region, frame_width: int, frame_height: int) -> Region:
    x, y, width, height = region
    clipped_x = max(0, min(x, frame_width))
    clipped_y = max(0, min(y, frame_height))
    clipped_width = max(0, min(width, frame_width - clipped_x))
    clipped_height = max(0, min(height, frame_height - clipped_y))
    return clipped_x, clipped_y, clipped_width, clipped_height


def _save_debug_snapshot(context: Context, name: str, image: Any) -> Path:
    if context.vision is not None:
        return context.vision.save_debug_snapshot(name, image)

    context.debug_snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = context.debug_snapshot_dir / f"{timestamp}_{name}.png"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write debug snapshot: {path}")
    return path


def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    context = build_standalone_context(config)

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(
            f"Warning: focusing game window. Starting align_to_nav_beacon in {STANDALONE_START_DELAY_SECONDS:.1f} seconds..."
        )
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    action = AlignToNavBeacon(
        config=NavBeaconAlignConfig(
            template_path=STANDALONE_TEMPLATE_PATH,
        )
    )
    result = action.run(context)
    print(result.success, result.reason)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
