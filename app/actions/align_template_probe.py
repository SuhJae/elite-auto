from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.actions.align import WINDOW_TITLE
from app.actions.align_support.detection import (
    _build_fallback_warm_mask,
    _build_marker_mask,
    _detect_compass_center_dot_candidates,
    _detect_compass_circle,
    _extract_roi,
    _recover_best_compass_circle_from_center_hints,
    _refine_compass_circle_with_outer_template,
    detect_compass_marker,
)
from app.actions.align_support.models import AlignConfig
from app.actions.navigation_ocr import _find_window_client_region, _resolve_output_region
from app.adapters.capture_dxcam import DxcamCapture


def _capture_live_frame() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    window_capture_region = _find_window_client_region(WINDOW_TITLE)
    if window_capture_region is None:
        raise RuntimeError(f"Could not locate the {WINDOW_TITLE!r} window.")
    output_index, capture_region = _resolve_output_region(window_capture_region)
    if output_index is None or capture_region is None:
        raise RuntimeError("Could not resolve the target display output.")

    capture = DxcamCapture(output_index=output_index, target_fps=60)
    capture.start()
    try:
        frame = capture.grab(region=capture_region)
    finally:
        capture.stop()
    if frame is None:
        raise RuntimeError("Capture returned no frame.")
    return frame, capture_region


def _draw_circle(roi: np.ndarray, circle: dict[str, float] | None, color: tuple[int, int, int], label: str) -> None:
    if circle is None:
        return
    center = (int(round(circle["center_x"])), int(round(circle["center_y"])))
    radius = max(1, int(round(circle["radius"])))
    cv2.circle(roi, center, radius, color, 1)
    cv2.drawMarker(roi, center, color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
    cv2.putText(
        roi,
        label,
        (center[0] + 4, center[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def _build_probe_images(frame: np.ndarray, config: AlignConfig) -> tuple[np.ndarray, np.ndarray]:
    roi_region = config.roi_region()
    roi = _extract_roi(frame, roi_region).copy()
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    marker_mask = _build_marker_mask(roi, hsv, config)
    closed_marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    fallback_mask = _build_fallback_warm_mask(hsv, config)
    center_candidates = _detect_compass_center_dot_candidates(closed_marker_mask, hsv, fallback_mask, config)
    center_hint = None if not center_candidates else (center_candidates[0]["x"], center_candidates[0]["y"])

    coarse_config = replace(config, outer_template_enabled=False)
    coarse_circle = _detect_compass_circle(
        roi,
        hsv,
        coarse_config,
        center_hint_x=None if center_hint is None else center_hint[0],
        center_hint_y=None if center_hint is None else center_hint[1],
    )
    best_recovery = _recover_best_compass_circle_from_center_hints(
        roi=roi,
        hsv=hsv,
        primary_mask=closed_marker_mask,
        fallback_mask=fallback_mask,
        config=config,
        center_dot_candidates=center_candidates,
    )

    template_refined = None
    if coarse_circle is not None:
        template_refined = _refine_compass_circle_with_outer_template(
            roi=roi,
            hsv=hsv,
            center_x=coarse_circle["center_x"],
            center_y=coarse_circle["center_y"],
            expected_radius=coarse_circle["radius"],
            config=config,
            max_center_error_px=max(8.0, config.outer_template_search_padding_px),
        )
    elif best_recovery is not None:
        recovery_circle = best_recovery["circle"]
        template_refined = _refine_compass_circle_with_outer_template(
            roi=roi,
            hsv=hsv,
            center_x=recovery_circle["center_x"],
            center_y=recovery_circle["center_y"],
            expected_radius=recovery_circle["radius"],
            config=config,
            max_center_error_px=max(8.0, config.outer_template_search_padding_px),
        )

    result = detect_compass_marker(frame, config)

    roi_overlay = roi.copy()
    for index, candidate in enumerate(center_candidates[:8]):
        point = (int(round(candidate["x"])), int(round(candidate["y"])))
        color = (0, 200, 255) if index == 0 else (0, 128, 200)
        cv2.circle(roi_overlay, point, 3, color, -1)
        cv2.putText(
            roi_overlay,
            f"dot{index}",
            (point[0] + 3, point[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    _draw_circle(roi_overlay, coarse_circle, (0, 215, 255), "coarse")
    _draw_circle(roi_overlay, None if best_recovery is None else best_recovery["circle"], (255, 180, 0), "recovery")
    _draw_circle(roi_overlay, template_refined, (0, 255, 0), "template")
    if result.marker is not None and result.marker.compass_center_x is not None and result.marker.compass_center_y is not None:
        final_circle = {
            "center_x": result.marker.compass_center_x - roi_region[0],
            "center_y": result.marker.compass_center_y - roi_region[1],
            "radius": result.marker.compass_radius_estimate_px or config.compass_radius_px,
        }
        _draw_circle(roi_overlay, final_circle, (255, 255, 255), "final")
        marker_local = (
            int(round(result.marker.marker_x - roi_region[0])),
            int(round(result.marker.marker_y - roi_region[1])),
        )
        marker_color = (0, 255, 0) if result.marker.is_filled else (0, 180, 255)
        cv2.circle(roi_overlay, marker_local, 3, marker_color, -1)
        cv2.line(
            roi_overlay,
            (int(round(final_circle["center_x"])), int(round(final_circle["center_y"]))),
            marker_local,
            marker_color,
            1,
        )

    lines = [
        f"result={result.status}",
        "coarse="
        + (
            "-"
            if coarse_circle is None
            else f"({coarse_circle['center_x']:.1f},{coarse_circle['center_y']:.1f}) r={coarse_circle['radius']:.1f}"
        ),
        "template="
        + (
            "-"
            if template_refined is None
            else (
                f"({template_refined['center_x']:.1f},{template_refined['center_y']:.1f}) "
                f"r={template_refined['radius']:.1f} "
                f"score={template_refined['template_score']:.3f}"
            )
        ),
        "warm_hint="
        + (
            "-"
            if center_hint is None
            else f"({center_hint[0]:.1f},{center_hint[1]:.1f}) count={len(center_candidates)}"
        ),
    ]
    y = 16
    for line in lines:
        cv2.putText(roi_overlay, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16

    frame_overlay = frame.copy()
    left, top, width, height = roi_region
    cv2.rectangle(frame_overlay, (left, top), (left + width, top + height), (255, 255, 255), 1)
    frame_overlay[top : top + height, left : left + width] = roi_overlay
    return frame_overlay, roi_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay or live-probe the template-refined compass detector.")
    parser.add_argument("--image", type=str, help="Path to a saved full-frame PNG to analyze.")
    parser.add_argument("--live", action="store_true", help="Capture one frame through the same dxcam path as align.py.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("debug_snapshots") / "template_probe"),
        help="Directory for saved overlay images.",
    )
    args = parser.parse_args()

    if not args.live and not args.image:
        parser.error("Provide --image or --live.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AlignConfig(debug_window_enabled=False)

    if args.live:
        frame, capture_region = _capture_live_frame()
        source_label = "live"
        print(f"Captured live frame from region={capture_region}")
    else:
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.image}")
        source_label = Path(args.image).stem

    frame_overlay, roi_overlay = _build_probe_images(frame, config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    full_path = output_dir / f"{timestamp}_{source_label}_template_probe_full.png"
    roi_path = output_dir / f"{timestamp}_{source_label}_template_probe_roi.png"
    if not cv2.imwrite(str(full_path), frame_overlay):
        raise RuntimeError(f"Failed to write overlay: {full_path}")
    if not cv2.imwrite(str(roi_path), roi_overlay):
        raise RuntimeError(f"Failed to write overlay: {roi_path}")

    print(full_path)
    print(roi_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
