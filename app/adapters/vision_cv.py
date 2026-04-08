from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from app.domain.models import TemplateMatch
from app.domain.protocols import Region, VisionSystem


class OpenCVVisionSystem(VisionSystem):
    """OpenCV-backed helper for localized image checks.

    TODO: Add OCR and richer HUD/template heuristics in later passes.
    """

    def __init__(self, debug_snapshot_dir: Path) -> None:
        self._debug_snapshot_dir = Path(debug_snapshot_dir)
        self._debug_snapshot_dir.mkdir(parents=True, exist_ok=True)

    def match_template(
        self,
        image: Any,
        template: Any,
        region: Region | None = None,
        threshold: float = 0.9,
    ) -> TemplateMatch | None:
        if image is None or template is None:
            raise ValueError("Image and template are required for template matching.")

        roi = _extract_region(image, region)
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, max_loc = cv2.minMaxLoc(result)

        if confidence < threshold:
            return None

        height, width = template.shape[:2]
        if region:
            max_loc = (max_loc[0] + region[0], max_loc[1] + region[1])

        return TemplateMatch(
            confidence=float(confidence),
            top_left=(int(max_loc[0]), int(max_loc[1])),
            size=(int(width), int(height)),
            region=region,
        )

    def save_debug_snapshot(self, name: str, image: Any) -> Path:
        if image is None:
            raise ValueError("Cannot save an empty debug snapshot.")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = self._debug_snapshot_dir / f"{timestamp}_{name}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write debug snapshot: {path}")
        return path


def _extract_region(image: Any, region: Region | None) -> Any:
    if region is None:
        return image
    x, y, width, height = region
    return image[y : y + height, x : x + width]
