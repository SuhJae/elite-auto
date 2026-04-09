from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.actions.align_support.models import AlignConfig, CompassMarker, CompassReadResult
from app.domain.protocols import Region

OUTER_TEMPLATE_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "compass_outer_template.png"


@lru_cache(maxsize=1)
def _load_outer_template_asset() -> tuple[np.ndarray, float, float, float] | None:
    template_rgba = cv2.imread(str(OUTER_TEMPLATE_ASSET_PATH), cv2.IMREAD_UNCHANGED)
    if template_rgba is None:
        return None
    if template_rgba.ndim == 2:
        alpha = template_rgba
    elif template_rgba.shape[2] >= 4:
        alpha = template_rgba[:, :, 3]
    else:
        alpha = cv2.cvtColor(template_rgba, cv2.COLOR_BGR2GRAY)

    template_mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(template_mask)
    if xs.size == 0 or ys.size == 0:
        return None

    center_x = (template_mask.shape[1] - 1) / 2.0
    center_y = (template_mask.shape[0] - 1) / 2.0
    distances = np.sqrt(((xs.astype(np.float32) - center_x) ** 2) + ((ys.astype(np.float32) - center_y) ** 2))
    nominal_radius = float(distances.max()) if distances.size else 0.0
    return template_mask, center_x, center_y, nominal_radius


@lru_cache(maxsize=64)
def _scaled_outer_template(scale: float) -> tuple[np.ndarray, float, float, float] | None:
    asset = _load_outer_template_asset()
    if asset is None:
        return None

    template_mask, center_x, center_y, nominal_radius = asset
    scale = float(scale)
    target_width = max(8, int(round(template_mask.shape[1] * scale)))
    target_height = max(8, int(round(template_mask.shape[0] * scale)))
    resized = cv2.resize(template_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    scale_x = target_width / float(template_mask.shape[1])
    scale_y = target_height / float(template_mask.shape[0])
    scaled_radius = nominal_radius * ((scale_x + scale_y) * 0.5)
    return resized, center_x * scale_x, center_y * scale_y, scaled_radius


def _outer_template_scale_values(config: AlignConfig) -> list[float]:
    scale_min = min(config.outer_template_scale_min, config.outer_template_scale_max)
    scale_max = max(config.outer_template_scale_min, config.outer_template_scale_max)
    if abs(scale_max - scale_min) < 1e-6:
        return [float(scale_min)]
    scale_step = max(0.01, float(config.outer_template_scale_step))
    values: list[float] = []
    current = scale_min
    while current <= scale_max + (scale_step * 0.5):
        values.append(round(current, 4))
        current += scale_step
    return values


def _preferred_outer_radius(config: AlignConfig) -> float:
    if config.outer_template_enabled:
        asset = _load_outer_template_asset()
        if asset is not None:
            return max(1.0, float(asset[3]))
    return max(1.0, float(config.compass_radius_px))


def _build_outer_template_match_image(
    roi: np.ndarray,
    strong_ring_mask: np.ndarray,
    weak_ring_mask: np.ndarray,
) -> np.ndarray:
    blue = roi[:, :, 0].astype(np.float32)
    green = roi[:, :, 1].astype(np.float32)
    red = roi[:, :, 2].astype(np.float32)
    blue_excess = np.clip(blue - (green * 0.55) - (red * 0.45), 0.0, None)
    max_blue_excess = float(blue_excess.max())
    if max_blue_excess > 1e-6:
        blue_excess = blue_excess / max_blue_excess
    else:
        blue_excess = np.zeros_like(blue_excess, dtype=np.float32)

    combined = (
        ((weak_ring_mask.astype(np.float32) / 255.0) * 0.55)
        + ((strong_ring_mask.astype(np.float32) / 255.0) * 0.30)
        + (blue_excess * 0.15)
    )
    match_image = np.clip(combined * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.GaussianBlur(match_image, (3, 3), 0)


def _refine_compass_circle_with_outer_template(
    *,
    roi: np.ndarray,
    hsv: np.ndarray,
    center_x: float,
    center_y: float,
    expected_radius: float,
    config: AlignConfig,
    strong_ring_mask: np.ndarray | None = None,
    weak_ring_mask: np.ndarray | None = None,
    max_center_error_px: float | None = None,
) -> dict[str, float] | None:
    if not config.outer_template_enabled:
        return None

    if strong_ring_mask is None or weak_ring_mask is None:
        strong_ring_mask, weak_ring_mask = _build_compass_ring_masks(roi, hsv, config, preserve_outer_ring=True)
    if np.count_nonzero(weak_ring_mask) == 0 and np.count_nonzero(strong_ring_mask) == 0:
        return None

    match_image = _build_outer_template_match_image(roi, strong_ring_mask, weak_ring_mask)
    search_padding = max(6, int(config.outer_template_search_padding_px))
    center_budget = max(
        float(search_padding),
        float(max_center_error_px if max_center_error_px is not None else search_padding),
    )
    span = max(expected_radius, 12.0) + search_padding + center_budget
    search_left = max(0, int(math.floor(center_x - span)))
    search_top = max(0, int(math.floor(center_y - span)))
    search_right = min(match_image.shape[1], int(math.ceil(center_x + span + 1.0)))
    search_bottom = min(match_image.shape[0], int(math.ceil(center_y + span + 1.0)))
    search_region = match_image[search_top:search_bottom, search_left:search_right]
    if search_region.size == 0:
        return None

    best: dict[str, float] | None = None
    best_score = float("-inf")
    expected_radius = max(1.0, float(expected_radius))

    for scale in _outer_template_scale_values(config):
        scaled = _scaled_outer_template(scale)
        if scaled is None:
            continue
        template_mask, offset_x, offset_y, template_radius = scaled
        if search_region.shape[0] < template_mask.shape[0] or search_region.shape[1] < template_mask.shape[1]:
            continue

        result = cv2.matchTemplate(search_region, template_mask, cv2.TM_CCORR_NORMED)
        _, template_score, _, max_loc = cv2.minMaxLoc(result)
        top_left_x = search_left + max_loc[0]
        top_left_y = search_top + max_loc[1]
        candidate_center_x = float(top_left_x + offset_x)
        candidate_center_y = float(top_left_y + offset_y)
        center_delta = math.hypot(candidate_center_x - center_x, candidate_center_y - center_y)
        if center_delta > (center_budget + 2.0):
            continue

        geometry_score = _score_compass_circle_candidate(
            strong_ring_mask=strong_ring_mask,
            weak_ring_mask=weak_ring_mask,
            center_x=candidate_center_x,
            center_y=candidate_center_y,
            radius=template_radius,
            axis_x=template_radius,
            axis_y=template_radius,
            angle_degrees=0.0,
            expected_center_x=center_x,
            expected_center_y=center_y,
            expected_radius=expected_radius,
            max_center_error_px=center_budget + 2.0,
            config=config,
        )
        if not math.isfinite(geometry_score):
            continue

        template_region = match_image[top_left_y : top_left_y + template_mask.shape[0], top_left_x : top_left_x + template_mask.shape[1]]
        template_on = template_mask > 0
        overlap = float(template_region[template_on].mean() / 255.0) if np.any(template_on) else 0.0
        score = (
            geometry_score
            + (float(template_score) * 3.2)
            + (overlap * 2.4)
            - (center_delta * 0.06)
            - (abs(template_radius - expected_radius) * 0.04)
        )
        if score > best_score:
            best_score = score
            best = {
                "center_x": candidate_center_x,
                "center_y": candidate_center_y,
                "radius": float(template_radius),
                "score": float(score),
                "template_score": float(template_score),
                "template_overlap": float(overlap),
                "template_top_left_x": float(top_left_x),
                "template_top_left_y": float(top_left_y),
                "template_scale": float(scale),
            }

    if best is None or best["template_score"] < config.outer_template_accept_score:
        return None
    return best


def _has_local_ring_support(
    *,
    strong_ring_mask: np.ndarray,
    weak_ring_mask: np.ndarray,
    center_x: float,
    center_y: float,
    config: AlignConfig,
) -> bool:
    span = max(10, int(round(config.compass_radius_px * 0.85)))
    left = max(0, int(math.floor(center_x - span)))
    top = max(0, int(math.floor(center_y - span)))
    right = min(weak_ring_mask.shape[1], int(math.ceil(center_x + span + 1.0)))
    bottom = min(weak_ring_mask.shape[0], int(math.ceil(center_y + span + 1.0)))
    if left >= right or top >= bottom:
        return False
    weak_patch = weak_ring_mask[top:bottom, left:right]
    strong_patch = strong_ring_mask[top:bottom, left:right]
    ring_pixels = int(np.count_nonzero(weak_patch)) + int(np.count_nonzero(strong_patch))
    return ring_pixels >= max(40, int(round(config.compass_radius_px * 2.0)))


def _ring_mask_edge_counts(ring_mask: np.ndarray, config: AlignConfig) -> dict[str, int]:
    edge_margin = max(8, int(round(config.compass_radius_px * 0.25)))
    return {
        "top": int(ring_mask[:edge_margin, :].sum()),
        "bottom": int(ring_mask[-edge_margin:, :].sum()),
        "left": int(ring_mask[:, :edge_margin].sum()),
        "right": int(ring_mask[:, -edge_margin:].sum()),
    }


def _ring_mask_is_likely_clipped(ring_mask: np.ndarray, config: AlignConfig) -> bool:
    if not np.any(ring_mask):
        return False
    edge_counts = _ring_mask_edge_counts(ring_mask, config)
    return any(count >= max(24, int(sum(edge_counts.values()) * 0.18)) for count in edge_counts.values())


def _is_finite_circle_candidate(candidate: dict[str, float]) -> bool:
    return all(
        math.isfinite(float(candidate.get(key, 0.0)))
        for key in ("center_x", "center_y", "radius", "axis_x", "axis_y", "angle_degrees", "source_bonus")
    )


def _has_warm_content_inside_circle(
    primary_mask: np.ndarray,
    fallback_mask: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    search_radius_px: float,
) -> bool:
    """Return True if any warm-colored pixel exists within search_radius_px of the given center.

    The compass ring is the only blue ring in the cockpit that consistently
    contains a warm (orange/amber) marker inside it. Panel borders and the
    console dome ring have no such content at their center positions.
    """
    height, width = primary_mask.shape[:2]
    cx = int(round(center_x))
    cy = int(round(center_y))
    r = int(math.ceil(search_radius_px))
    x0 = max(0, cx - r)
    y0 = max(0, cy - r)
    x1 = min(width, cx + r + 1)
    y1 = min(height, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    patch_primary = primary_mask[y0:y1, x0:x1]
    if np.any(patch_primary > 0):
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
        within = dist_sq <= search_radius_px ** 2
        if np.any(np.logical_and(within, patch_primary > 0)):
            return True
    patch_fallback = fallback_mask[y0:y1, x0:x1]
    if np.any(patch_fallback > 0):
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
        within = dist_sq <= search_radius_px ** 2
        if np.any(np.logical_and(within, patch_fallback > 0)):
            return True
    return False


def detect_compass_marker(image: Any, config: AlignConfig) -> CompassReadResult:
    if image is None:
        return CompassReadResult(status="missing")

    primary_region = config.roi_region()
    result = _detect_compass_marker_in_region(image, primary_region, config)
    if result.is_detected:
        return result
    if result.status not in {"missing", "circle_only"}:
        return result

    for fallback_region in _fallback_compass_regions(image, primary_region, config):
        fallback_result = _detect_compass_marker_in_region(image, fallback_region, config)
        if fallback_result.is_detected:
            return fallback_result
        if result.status == "missing" and fallback_result.status == "circle_only":
            result = fallback_result

    return result


def _detect_compass_marker_in_region(
    image: Any,
    roi_region: Region,
    config: AlignConfig,
) -> CompassReadResult:
    roi = _extract_roi(image, roi_region)
    return _detect_compass_marker_in_roi(roi, roi_region, config)


def _detect_compass_marker_in_roi(
    roi: np.ndarray,
    roi_region: Region,
    config: AlignConfig,
) -> CompassReadResult:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    strong_ring_mask, weak_ring_mask = _build_compass_ring_masks(roi, hsv, config)
    ring_mask = np.logical_or(strong_ring_mask > 0, weak_ring_mask > 0)
    marker_mask = _build_marker_mask(roi, hsv, config)
    mask_closed = cv2.morphologyEx(marker_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    fallback_mask = _build_fallback_warm_mask(hsv, config)

    # Phase 1: detect warm center-dot candidates independently. The real compass
    # behaves like a small cyan bubble that contains a warm marker, so we keep
    # several plausible warm candidates around instead of trusting a single
    # center-biased hint.
    center_dot_candidates = _detect_compass_center_dot_candidates(mask_closed, hsv, fallback_mask, config)
    center_dot_hint = None if not center_dot_candidates else (
        center_dot_candidates[0]["x"],
        center_dot_candidates[0]["y"],
    )
    if not center_dot_candidates and _ring_mask_is_likely_clipped(ring_mask, config):
        return CompassReadResult(status="missing")
    best_recovery: dict[str, Any] | None = None

    def compute_best_recovery() -> dict[str, Any] | None:
        nonlocal best_recovery
        if best_recovery is None:
            best_recovery = _recover_best_compass_circle_from_center_hints(
                roi=roi,
                hsv=hsv,
                primary_mask=mask_closed,
                fallback_mask=fallback_mask,
                config=config,
                center_dot_candidates=center_dot_candidates,
                strong_ring_mask=strong_ring_mask,
                weak_ring_mask=weak_ring_mask,
            )
        return best_recovery

    # Phase 2: detect the compass circle independently of the dot hint.
    circle = _detect_compass_circle(
        roi,
        hsv,
        config,
        center_hint_x=None if center_dot_hint is None else center_dot_hint[0],
        center_hint_y=None if center_dot_hint is None else center_dot_hint[1],
    )
    recovered_marker_detection: dict[str, float | str] | None = None
    if circle is None:
        recovery = compute_best_recovery()
        if recovery is None:
            return CompassReadResult(status="missing")
        circle = recovery["circle"]
        recovered_marker_detection = recovery["marker_detection"]

    compass_center_local_x = circle["center_x"]
    compass_center_local_y = circle["center_y"]
    compass_radius_estimate = circle["radius"]

    # Validate: the compass ring is the only blue ring in the cockpit that
    # contains warm (orange) marker content inside it. Reject circles that
    # have no warm pixels within compass_radius_px + small margin of their
    # center — those are console-ring arcs or panel borders, not the compass.
    if not _has_warm_content_inside_circle(
        mask_closed,
        fallback_mask,
        center_x=compass_center_local_x,
        center_y=compass_center_local_y,
        search_radius_px=config.max_marker_distance_px,
    ):
        recovery = compute_best_recovery()
        if recovery is not None:
            circle = recovery["circle"]
            recovered_marker_detection = recovery["marker_detection"]
            compass_center_local_x = circle["center_x"]
            compass_center_local_y = circle["center_y"]
            compass_radius_estimate = circle["radius"]
        else:
            return CompassReadResult(status="circle_only")

    # Phase 3: detect the marker relative to the finalized circle center.
    marker_detection = (
        recovered_marker_detection
        if recovered_marker_detection is not None
        else _detect_compass_marker_relative_to_circle(
            primary_mask=mask_closed,
            fallback_mask=fallback_mask,
            hsv=hsv,
            config=config,
            local_center_x=compass_center_local_x,
            local_center_y=compass_center_local_y,
        )
    )
    suspicious_hint_distance = None
    if center_dot_hint is not None:
        suspicious_hint_distance = math.hypot(
            compass_center_local_x - center_dot_hint[0],
            compass_center_local_y - center_dot_hint[1],
        )
    should_try_recovery_comparison = (
        marker_detection["status"] != "detected"
        or (
            suspicious_hint_distance is not None
            and suspicious_hint_distance > min(
                config.max_marker_distance_px,
                compass_radius_estimate + max(6.0, config.center_dot_circle_center_max_distance_px),
            )
        )
    )
    if recovered_marker_detection is None and should_try_recovery_comparison:
        recovery = compute_best_recovery()
    else:
        recovery = None
    if recovered_marker_detection is None and recovery is not None:
        recovery_detection = recovery["marker_detection"]
        if _should_prefer_hint_recovery(marker_detection, recovery_detection):
            circle = recovery["circle"]
            recovered_marker_detection = recovery_detection
            marker_detection = recovery_detection
            compass_center_local_x = circle["center_x"]
            compass_center_local_y = circle["center_y"]
            compass_radius_estimate = circle["radius"]
    if marker_detection["status"] == "circle_only":
        return CompassReadResult(status="circle_only")
    if marker_detection["status"] != "detected":
        return CompassReadResult(status=str(marker_detection["status"]))

    state = str(marker_detection["marker_state"])
    marker_local_x = float(marker_detection["marker_x"])
    marker_local_y = float(marker_detection["marker_y"])
    inner_occupancy = float(marker_detection["inner_occupancy"])
    outer_ring_occupancy = float(marker_detection["outer_ring_occupancy"])
    candidate_area = int(marker_detection["area"])

    marker_x = roi_region[0] + marker_local_x
    marker_y = roi_region[1] + marker_local_y
    compass_center_x = roi_region[0] + compass_center_local_x
    compass_center_y = roi_region[1] + compass_center_local_y
    dx = marker_x - compass_center_x
    dy = marker_y - compass_center_y
    control_dx, control_dy = _rotate_offset(dx, dy, config.compass_control_rotation_radians)
    distance = math.hypot(dx, dy)
    normalized_radius = min(1.0, distance / max(compass_radius_estimate, 1.0))
    front_semisphere_radians = math.asin(normalized_radius)
    if state == "filled":
        target_off_boresight_radians = front_semisphere_radians
        phase_adjustment_radians = front_semisphere_radians
    else:
        target_off_boresight_radians = math.pi - front_semisphere_radians
        phase_adjustment_radians = max(0.0, (math.pi / 2.0) - front_semisphere_radians)

    marker = CompassMarker(
        marker_state=state,
        marker_x=marker_x,
        marker_y=marker_y,
        dx=dx,
        dy=dy,
        control_dx=control_dx,
        control_dy=control_dy,
        distance=distance,
        normalized_radius=normalized_radius,
        front_semisphere_radians=front_semisphere_radians,
        target_off_boresight_radians=target_off_boresight_radians,
        phase_adjustment_radians=phase_adjustment_radians,
        component_area=candidate_area,
        inner_occupancy=inner_occupancy,
        outer_ring_occupancy=outer_ring_occupancy,
        roi_region=roi_region,
        compass_center_x=compass_center_x,
        compass_center_y=compass_center_y,
        compass_radius_estimate_px=compass_radius_estimate,
    )
    return CompassReadResult(status=state, marker=marker)


def _fallback_compass_regions(
    image: Any,
    primary_region: Region,
    config: AlignConfig,
) -> list[Region]:
    try:
        roi = _extract_roi(image, primary_region)
    except ValueError:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    strong_ring_mask, weak_ring_mask = _build_compass_ring_masks(roi, hsv, config)
    ring_mask = np.logical_or(strong_ring_mask > 0, weak_ring_mask > 0)
    if not np.any(ring_mask):
        return []
    edge_counts = _ring_mask_edge_counts(ring_mask, config)

    regions: list[Region] = []
    if edge_counts["bottom"] >= max(24, int(edge_counts["top"] * 1.5)):
        for dy in (24, 48):
            region = _shift_region(primary_region, image.shape, dx=0, dy=dy)
            if region is not None:
                regions.append(region)
    elif edge_counts["top"] >= max(24, int(edge_counts["bottom"] * 1.5)):
        for dy in (-24, -48):
            region = _shift_region(primary_region, image.shape, dx=0, dy=dy)
            if region is not None:
                regions.append(region)

    if edge_counts["left"] >= max(24, int(edge_counts["right"] * 1.5)):
        for dx in (-16,):
            region = _shift_region(primary_region, image.shape, dx=dx, dy=0)
            if region is not None:
                regions.append(region)
    elif edge_counts["right"] >= max(24, int(edge_counts["left"] * 1.5)):
        for dx in (16,):
            region = _shift_region(primary_region, image.shape, dx=dx, dy=0)
            if region is not None:
                regions.append(region)

    deduped: list[Region] = []
    seen: set[Region] = set()
    for region in regions:
        if region == primary_region or region in seen:
            continue
        seen.add(region)
        deduped.append(region)
    return deduped


def _is_recovery_candidate_likely_clipped(
    *,
    strong_ring_mask: np.ndarray,
    weak_ring_mask: np.ndarray,
    center_x: float,
    center_y: float,
    config: AlignConfig,
) -> bool:
    ring_mask = np.logical_or(strong_ring_mask > 0, weak_ring_mask > 0)
    if not np.any(ring_mask):
        return False

    height, width = ring_mask.shape[:2]
    hint_margin = max(12, int(round(config.compass_radius_px * 0.60)))
    edge_margin = max(8, int(round(config.compass_radius_px * 0.35)))
    edge_threshold = max(20, int(round(config.compass_radius_px * 1.5)))

    if center_x < hint_margin and int(ring_mask[:, :edge_margin].sum()) >= edge_threshold:
        return True
    if center_x > (width - 1 - hint_margin) and int(ring_mask[:, -edge_margin:].sum()) >= edge_threshold:
        return True
    if center_y < hint_margin and int(ring_mask[:edge_margin, :].sum()) >= edge_threshold:
        return True
    if center_y > (height - 1 - hint_margin) and int(ring_mask[-edge_margin:, :].sum()) >= edge_threshold:
        return True
    return False


def _shift_region(region: Region, image_shape: tuple[int, ...], *, dx: int, dy: int) -> Region | None:
    left, top, width, height = region
    shifted_left = left + dx
    shifted_top = top + dy
    if shifted_left < 0 or shifted_top < 0:
        return None
    if shifted_left + width > image_shape[1] or shifted_top + height > image_shape[0]:
        return None
    return (shifted_left, shifted_top, width, height)


def _recover_compass_circle_from_center_hint(
    *,
    roi: np.ndarray,
    hsv: np.ndarray,
    primary_mask: np.ndarray,
    fallback_mask: np.ndarray,
    config: AlignConfig,
    center_hint_x: float,
    center_hint_y: float,
    strong_ring_mask: np.ndarray | None = None,
    weak_ring_mask: np.ndarray | None = None,
    validate_marker: bool = True,
) -> dict[str, Any] | None:
    if strong_ring_mask is None or weak_ring_mask is None:
        strong_ring_mask, weak_ring_mask = _build_compass_ring_masks(roi, hsv, config)
    if not _has_local_ring_support(
        strong_ring_mask=strong_ring_mask,
        weak_ring_mask=weak_ring_mask,
        center_x=center_hint_x,
        center_y=center_hint_y,
        config=config,
    ):
        return None
    refined_center = _find_inner_ring_center_by_offset_search(
        strong_ring_mask,
        expected_center_x=center_hint_x,
        expected_center_y=center_hint_y,
        config=config,
        max_center_error_px=max(config.inner_ring_center_search_radius_px, config.center_dot_search_radius_px),
    )
    if refined_center is None:
        return None

    refined_center_x, refined_center_y, refined_score = refined_center
    if math.hypot(refined_center_x - center_hint_x, refined_center_y - center_hint_y) > max(
        12.0,
        config.center_dot_search_radius_px,
        config.compass_radius_px * 0.70,
    ):
        return None

    final_center_x = refined_center_x
    final_center_y = refined_center_y
    final_score = refined_score
    refined_radius = _refine_compass_radius_around_center(
        strong_ring_mask=strong_ring_mask,
        weak_ring_mask=weak_ring_mask,
        center_x=refined_center_x,
        center_y=refined_center_y,
        config=config,
    )
    template_refined = _refine_compass_circle_with_outer_template(
        roi=roi,
        hsv=hsv,
        center_x=refined_center_x,
        center_y=refined_center_y,
        expected_radius=refined_radius,
        config=config,
        strong_ring_mask=strong_ring_mask,
        weak_ring_mask=weak_ring_mask,
        max_center_error_px=max(config.center_dot_circle_center_max_distance_px + 4.0, 10.0),
    )
    if template_refined is not None:
        final_center_x = float(template_refined["center_x"])
        final_center_y = float(template_refined["center_y"])
        refined_radius = float(template_refined["radius"])
        final_score = max(final_score, float(template_refined["score"]))

    if not _has_warm_content_inside_circle(
        primary_mask,
        fallback_mask,
        center_x=final_center_x,
        center_y=final_center_y,
        search_radius_px=config.max_marker_distance_px,
    ):
        return None

    marker_detection: dict[str, float | str] | None = None
    if validate_marker:
        marker_detection = _detect_compass_marker_relative_to_circle(
            primary_mask=primary_mask,
            fallback_mask=fallback_mask,
            hsv=hsv,
            config=config,
            local_center_x=final_center_x,
            local_center_y=final_center_y,
        )
        if marker_detection.get("status") != "detected":
            return None

    return {
        "circle": {
            "center_x": final_center_x,
            "center_y": final_center_y,
            "radius": refined_radius,
            "score": final_score,
        },
        "marker_detection": marker_detection,
        "center_hint_x": float(center_hint_x),
        "center_hint_y": float(center_hint_y),
    }


def _refine_compass_radius_around_center(
    *,
    strong_ring_mask: np.ndarray,
    weak_ring_mask: np.ndarray,
    center_x: float,
    center_y: float,
    config: AlignConfig,
) -> float:
    preferred_radius = _preferred_outer_radius(config)
    if config.outer_template_enabled:
        return preferred_radius

    radius_tolerance = max(2.0, float(config.circle_detection_radius_tolerance_px + 4.0))
    min_radius = max(14.0, preferred_radius - radius_tolerance)
    max_radius = preferred_radius + radius_tolerance
    best_radius = preferred_radius
    best_score = float("-inf")

    for radius in np.arange(min_radius, max_radius + 0.5, 1.0):
        score = _score_compass_circle_candidate(
            strong_ring_mask=strong_ring_mask,
            weak_ring_mask=weak_ring_mask,
            center_x=center_x,
            center_y=center_y,
            radius=float(radius),
            axis_x=float(radius),
            axis_y=float(radius),
            angle_degrees=0.0,
            expected_center_x=center_x,
            expected_center_y=center_y,
            expected_radius=float(radius),
            max_center_error_px=None,
            config=config,
        )
        if score > best_score:
            best_score = score
            best_radius = float(radius)

    return best_radius


def _recover_best_compass_circle_from_center_hints(
    *,
    roi: np.ndarray,
    hsv: np.ndarray,
    primary_mask: np.ndarray,
    fallback_mask: np.ndarray,
    config: AlignConfig,
    center_dot_candidates: list[dict[str, float]],
    strong_ring_mask: np.ndarray | None = None,
    weak_ring_mask: np.ndarray | None = None,
) -> dict[str, Any] | None:
    best_recovery: dict[str, Any] | None = None
    best_score = float("-inf")
    if not center_dot_candidates:
        return None

    if strong_ring_mask is None or weak_ring_mask is None:
        strong_ring_mask, weak_ring_mask = _build_compass_ring_masks(roi, hsv, config)
    coarse_recoveries: list[dict[str, Any]] = []
    for candidate in center_dot_candidates[: max(1, int(config.center_dot_max_candidates))]:
        if _is_recovery_candidate_likely_clipped(
            strong_ring_mask=strong_ring_mask,
            weak_ring_mask=weak_ring_mask,
            center_x=float(candidate["x"]),
            center_y=float(candidate["y"]),
            config=config,
        ):
            continue
        recovery = _recover_compass_circle_from_center_hint(
            roi=roi,
            hsv=hsv,
            primary_mask=primary_mask,
            fallback_mask=fallback_mask,
            config=config,
            center_hint_x=float(candidate["x"]),
            center_hint_y=float(candidate["y"]),
            strong_ring_mask=strong_ring_mask,
            weak_ring_mask=weak_ring_mask,
            validate_marker=False,
        )
        if recovery is None:
            continue
        score = float(recovery["circle"]["score"]) - float(candidate["distance"]) * 0.01
        recovery["_candidate_score"] = score
        coarse_recoveries.append(recovery)

    coarse_recoveries.sort(key=lambda item: float(item["_candidate_score"]), reverse=True)
    for recovery in coarse_recoveries[:2]:
        circle = recovery["circle"]
        marker_detection = _detect_compass_marker_relative_to_circle(
            primary_mask=primary_mask,
            fallback_mask=fallback_mask,
            hsv=hsv,
            config=config,
            local_center_x=float(circle["center_x"]),
            local_center_y=float(circle["center_y"]),
        )
        if marker_detection.get("status") != "detected":
            continue
        recovery["marker_detection"] = marker_detection
        score = _marker_detection_preference_score(marker_detection) + float(circle["score"])
        if score > best_score:
            best_score = score
            best_recovery = recovery

    return best_recovery


def _detect_compass_marker_relative_to_circle(
    *,
    primary_mask: np.ndarray,
    fallback_mask: np.ndarray,
    hsv: np.ndarray,
    config: AlignConfig,
    local_center_x: float,
    local_center_y: float,
) -> dict[str, float | str]:
    mask = primary_mask
    candidates = _candidate_components(
        primary_mask,
        hsv,
        config,
        local_center_x=local_center_x,
        local_center_y=local_center_y,
    )
    if not candidates:
        mask = fallback_mask
        candidates = _candidate_components(
            fallback_mask,
            hsv,
            config,
            local_center_x=local_center_x,
            local_center_y=local_center_y,
        )
    if not candidates:
        return {"status": "circle_only"}

    best_detection: dict[str, float | str] | None = None
    best_score = float("-inf")
    saw_ambiguous = False

    for candidate in candidates:
        centroid_x = candidate["centroid_x"]
        centroid_y = candidate["centroid_y"]
        mean_value = candidate["mean_value"]
        distance_to_circle = math.hypot(centroid_x - local_center_x, centroid_y - local_center_y)
        inner_occupancy, outer_ring_occupancy = _occupancy_scores(
            mask,
            centroid_x,
            centroid_y,
            config,
        )
        definitive_filled = (
            inner_occupancy >= config.filled_inner_occupancy_threshold
            and inner_occupancy >= config.definitive_filled_inner_occupancy_threshold
        )
        if (
            definitive_filled
        ):
            filled_score = (
                (inner_occupancy * 6.0)
                + (outer_ring_occupancy * 1.5)
                - (distance_to_circle * 0.05)
                + (mean_value * 0.01)
            )
            if filled_score > best_score:
                best_score = filled_score
                best_detection = {
                    "status": "detected",
                    "marker_state": "filled",
                    "marker_x": centroid_x,
                    "marker_y": centroid_y,
                    "inner_occupancy": inner_occupancy,
                    "outer_ring_occupancy": outer_ring_occupancy,
                    "area": candidate["area"],
                }
            continue

        refined = _refine_hollow_center(mask, centroid_x, centroid_y, config)
        if refined is not None:
            refined_distance = math.hypot(centroid_x - local_center_x, centroid_y - local_center_y)
            hollow_score = (
                (refined["outer_ring_occupancy"] * 7.0)
                - (refined["inner_occupancy"] * 5.5)
                - (refined_distance * 0.05)
                + (mean_value * 0.01)
                + 0.5
            )
            if hollow_score > best_score:
                best_score = hollow_score
                best_detection = {
                    "status": "detected",
                    "marker_state": "hollow",
                    "marker_x": centroid_x,
                    "marker_y": centroid_y,
                    "inner_occupancy": refined["inner_occupancy"],
                    "outer_ring_occupancy": refined["outer_ring_occupancy"],
                    "area": candidate["area"],
                }
        elif inner_occupancy >= config.filled_inner_occupancy_threshold:
            filled_score = (
                (inner_occupancy * 5.5)
                + (outer_ring_occupancy * 1.5)
                - (distance_to_circle * 0.05)
                + (mean_value * 0.01)
            )
            if filled_score > best_score:
                best_score = filled_score
                best_detection = {
                    "status": "detected",
                    "marker_state": "filled",
                    "marker_x": centroid_x,
                    "marker_y": centroid_y,
                    "inner_occupancy": inner_occupancy,
                    "outer_ring_occupancy": outer_ring_occupancy,
                    "area": candidate["area"],
                }
        elif outer_ring_occupancy > 0.0:
            saw_ambiguous = True

    if best_detection is not None:
        return best_detection
    if saw_ambiguous:
        return {"status": "ambiguous"}
    return {"status": "circle_only"}


def _marker_detection_preference_score(marker_detection: dict[str, float | str]) -> float:
    if marker_detection.get("status") != "detected":
        return float("-inf")

    state = str(marker_detection.get("marker_state"))
    inner_occupancy = float(marker_detection.get("inner_occupancy", 0.0))
    outer_ring_occupancy = float(marker_detection.get("outer_ring_occupancy", 0.0))
    area = float(marker_detection.get("area", 0.0))

    score = (outer_ring_occupancy * 4.0) + (area * 0.02)
    if state == "filled":
        score += 4.0 + (inner_occupancy * 3.0)
    else:
        score += 0.5 - (inner_occupancy * 3.0)
    return score


def _should_prefer_hint_recovery(
    current_detection: dict[str, float | str],
    recovery_detection: dict[str, float | str],
) -> bool:
    if current_detection.get("status") != "detected":
        return True
    return _marker_detection_preference_score(recovery_detection) > (
        _marker_detection_preference_score(current_detection) + 0.75
    )


def _build_marker_mask(roi: np.ndarray, hsv: np.ndarray, config: AlignConfig) -> np.ndarray:
    warm_mask = cv2.inRange(
        hsv,
        np.array([config.warm_h_min, config.warm_s_min, config.warm_v_min], dtype=np.uint8),
        np.array([config.warm_h_max, 255, 255], dtype=np.uint8),
    )

    blue_channel, green_channel, red_channel = cv2.split(roi)
    blue_minus_red = blue_channel.astype(np.int16) - red_channel.astype(np.int16)
    green_minus_red = green_channel.astype(np.int16) - red_channel.astype(np.int16)
    pale_mask = cv2.inRange(
        hsv,
        np.array([0, 0, config.pale_v_min], dtype=np.uint8),
        np.array([config.pale_h_max, config.pale_s_max, 255], dtype=np.uint8),
    )
    pale_mask = np.logical_and(
        pale_mask > 0,
        np.logical_and.reduce(
            (
                red_channel >= config.pale_red_min,
                green_channel >= config.pale_green_min,
                blue_minus_red <= config.pale_blue_red_delta_max,
                green_minus_red <= config.pale_green_red_delta_max,
            )
        ),
    )

    return np.where(np.logical_or(warm_mask > 0, pale_mask), 255, 0).astype(np.uint8)


def _build_fallback_warm_mask(hsv: np.ndarray, config: AlignConfig) -> np.ndarray:
    return cv2.inRange(
        hsv,
        np.array([0, config.fallback_warm_s_min, config.fallback_warm_v_min], dtype=np.uint8),
        np.array([config.warm_h_max, 255, 255], dtype=np.uint8),
    )


def _detect_compass_center_dot(
    mask: np.ndarray,
    hsv: np.ndarray,
    fallback_mask: np.ndarray,
    config: AlignConfig,
) -> tuple[float, float] | None:
    candidates = _detect_compass_center_dot_candidates(mask, hsv, fallback_mask, config)
    best_candidate = None if not candidates else candidates[0]
    if best_candidate is None:
        return None
    return best_candidate["x"], best_candidate["y"]


def _detect_compass_center_dot_candidates(
    mask: np.ndarray,
    hsv: np.ndarray,
    fallback_mask: np.ndarray,
    config: AlignConfig,
) -> list[dict[str, float]]:
    anchor_x = mask.shape[1] / 2.0
    anchor_y = mask.shape[0] / 2.0
    expanded_search_radius = max(mask.shape[0], mask.shape[1]) / 1.6
    candidates: list[dict[str, float]] = []

    candidates.extend(
        _find_center_dot_candidates(
            mask,
            hsv,
            anchor_x,
            anchor_y,
            config,
            search_radius_px=config.center_dot_search_radius_px,
        )
    )
    candidates.extend(
        _find_center_dot_candidates(
            fallback_mask,
            hsv,
            anchor_x,
            anchor_y,
            config,
            search_radius_px=config.center_dot_search_radius_px,
        )
    )
    if expanded_search_radius > config.center_dot_search_radius_px:
        candidates.extend(
            _find_center_dot_candidates(
                mask,
                hsv,
                anchor_x,
                anchor_y,
                config,
                search_radius_px=expanded_search_radius,
            )
        )
        candidates.extend(
            _find_center_dot_candidates(
                fallback_mask,
                hsv,
                anchor_x,
                anchor_y,
                config,
                search_radius_px=expanded_search_radius,
            )
        )

    deduped: list[dict[str, float]] = []
    seen: set[tuple[int, int]] = set()
    for candidate in sorted(candidates, key=lambda item: item["priority"], reverse=True):
        key = (int(round(candidate["x"] / 2.0)), int(round(candidate["y"] / 2.0)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= max(1, int(config.center_dot_max_candidates)):
            break
    return deduped


def _find_center_dot_candidate(
    mask: np.ndarray,
    hsv: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    config: AlignConfig,
    *,
    search_radius_px: float,
) -> dict[str, float] | None:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best_candidate: dict[str, float] | None = None
    best_key: tuple[float, float, float] | None = None

    for index in range(1, num_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < config.center_dot_min_area or area > config.center_dot_max_area:
            continue

        centroid_x = float(centroids[index][0])
        centroid_y = float(centroids[index][1])
        distance = math.hypot(centroid_x - anchor_x, centroid_y - anchor_y)
        if distance > search_radius_px:
            continue

        component_mask = labels == index
        mean_value = float(hsv[:, :, 2][component_mask].mean()) if np.any(component_mask) else 0.0
        distance_weight = distance if search_radius_px <= config.center_dot_search_radius_px else (distance * 0.35)
        candidate = {
            "x": centroid_x,
            "y": centroid_y,
            "distance": distance,
            "area": float(area),
            "mean_value": mean_value,
        }
        candidate_key = (distance_weight, -mean_value, -float(area))
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_candidate = candidate

    return best_candidate


def _find_center_dot_candidates(
    mask: np.ndarray,
    hsv: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    config: AlignConfig,
    *,
    search_radius_px: float,
) -> list[dict[str, float]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates: list[dict[str, float]] = []

    for index in range(1, num_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < config.center_dot_min_area or area > config.center_dot_max_area:
            continue

        centroid_x = float(centroids[index][0])
        centroid_y = float(centroids[index][1])
        distance = math.hypot(centroid_x - anchor_x, centroid_y - anchor_y)
        if distance > search_radius_px:
            continue

        component_mask = labels == index
        mean_value = float(hsv[:, :, 2][component_mask].mean()) if np.any(component_mask) else 0.0
        distance_weight = distance if search_radius_px <= config.center_dot_search_radius_px else (distance * 0.20)
        priority = (mean_value * 0.04) + (float(area) * 0.15) - distance_weight
        candidates.append(
            {
                "x": centroid_x,
                "y": centroid_y,
                "distance": distance,
                "area": float(area),
                "mean_value": mean_value,
                "priority": priority,
            }
        )

    return candidates


def _circle_search_bounds(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    expected_radius: float,
    config: AlignConfig,
    *,
    max_center_error_px: float | None = None,
) -> tuple[int, int, int, int]:
    center_tolerance = (
        max(0.0, max_center_error_px)
        if max_center_error_px is not None
        else max(0.0, config.circle_detection_center_hint_tolerance_px)
    )
    half_span = int(
        math.ceil(
            expected_radius
            + center_tolerance
            + max(0.0, config.circle_detection_crop_padding_px)
        )
    )
    left = max(0, int(math.floor(center_x - half_span)))
    top = max(0, int(math.floor(center_y - half_span)))
    right = min(width, int(math.ceil(center_x + half_span + 1.0)))
    bottom = min(height, int(math.ceil(center_y + half_span + 1.0)))
    return left, top, right, bottom


def _build_prototype_mask(
    roi: np.ndarray,
    prototype_bgr: tuple[float, float, float],
    distance_threshold: float,
) -> np.ndarray:
    prototype = np.array(prototype_bgr, dtype=np.float32)
    diff = roi.astype(np.float32) - prototype
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    return np.where(distance <= distance_threshold, 255, 0).astype(np.uint8)


def _build_compass_ring_masks(
    roi: np.ndarray,
    hsv: np.ndarray,
    config: AlignConfig,
    *,
    preserve_outer_ring: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    strong_hsv_mask = cv2.inRange(
        hsv,
        np.array([config.ring_blue_h_min, config.ring_blue_s_min, config.ring_blue_v_min], dtype=np.uint8),
        np.array([config.ring_blue_h_max, 255, 255], dtype=np.uint8),
    )
    weak_hsv_mask = cv2.inRange(
        hsv,
        np.array([config.outer_ring_blue_h_min, config.outer_ring_blue_s_min, config.outer_ring_blue_v_min], dtype=np.uint8),
        np.array([config.outer_ring_blue_h_max, 255, 255], dtype=np.uint8),
    )
    strong_proto_mask = _build_prototype_mask(
        roi,
        config.inner_ring_bgr_prototype,
        config.inner_ring_bgr_distance_threshold,
    )
    weak_proto_mask = _build_prototype_mask(
        roi,
        config.outer_ring_bgr_prototype,
        config.outer_ring_bgr_distance_threshold,
    )

    kernel = np.ones((3, 3), dtype=np.uint8)
    strong_ring_mask = np.where(
        np.logical_or(strong_hsv_mask > 0, strong_proto_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    weak_ring_mask = np.where(
        np.logical_or.reduce((weak_hsv_mask > 0, weak_proto_mask > 0, strong_ring_mask > 0)),
        255,
        0,
    ).astype(np.uint8)
    strong_ring_mask = cv2.morphologyEx(strong_ring_mask, cv2.MORPH_OPEN, kernel)
    strong_ring_mask = cv2.morphologyEx(strong_ring_mask, cv2.MORPH_CLOSE, kernel)
    weak_ring_mask = cv2.morphologyEx(weak_ring_mask, cv2.MORPH_OPEN, kernel)
    weak_ring_mask = cv2.morphologyEx(weak_ring_mask, cv2.MORPH_CLOSE, kernel)
    if config.inner_ring_tracking_only and not preserve_outer_ring:
        # When tracking only the brighter inner ring, the dim outer ring is more likely
        # to be corrupted by shadows and background detail than to add useful evidence.
        weak_ring_mask = strong_ring_mask.copy()
    return strong_ring_mask, weak_ring_mask


@lru_cache(maxsize=16)
def _inner_ring_template(
    *,
    radius: int,
    half_width: int,
    padding: int,
) -> tuple[np.ndarray, int, int]:
    center = radius + half_width + padding
    size = (center * 2) + 1
    yy, xx = np.ogrid[:size, :size]
    distances = np.sqrt(((xx - center) ** 2) + ((yy - center) ** 2))
    template = np.where(
        np.logical_and(distances >= (radius - half_width), distances <= (radius + half_width)),
        255,
        0,
    ).astype(np.uint8)
    return template, center, center


@lru_cache(maxsize=16)
def _inner_ring_search_offsets(
    *,
    radius: int,
    half_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    span = radius + half_width + 2
    yy, xx = np.mgrid[-span : span + 1, -span : span + 1]
    distances = np.sqrt((xx.astype(np.float32) ** 2) + (yy.astype(np.float32) ** 2))
    ring_mask = np.logical_and(distances >= (radius - half_width), distances <= (radius + half_width))
    center_mask = distances <= max(1.0, radius - 3.0)
    ring_y, ring_x = np.nonzero(ring_mask)
    center_y, center_x = np.nonzero(center_mask)
    origin = span
    return (
        ring_y.astype(np.int16) - origin,
        ring_x.astype(np.int16) - origin,
        center_y.astype(np.int16) - origin,
        center_x.astype(np.int16) - origin,
    )


def _find_inner_ring_center_by_offset_search(
    mask: np.ndarray,
    *,
    expected_center_x: float,
    expected_center_y: float,
    config: AlignConfig,
    max_center_error_px: float | None = None,
) -> tuple[float, float, float] | None:
    if mask.size == 0 or np.count_nonzero(mask) == 0:
        return None

    base_search_radius = max(1, int(config.inner_ring_center_search_radius_px))
    search_radius = (
        min(base_search_radius, max(1, int(math.ceil(max_center_error_px))))
        if max_center_error_px is not None
        else base_search_radius
    )
    half_width = max(1, int(round(config.ring_score_inner_width_px)))
    ring_y_offsets, ring_x_offsets, center_y_offsets, center_x_offsets = _inner_ring_search_offsets(
        radius=max(2, int(round(config.inner_ring_radius_px))),
        half_width=half_width,
    )

    best: tuple[float, float, float] | None = None
    best_score = float("-inf")
    expected_x = int(round(expected_center_x))
    expected_y = int(round(expected_center_y))
    height, width = mask.shape[:2]

    for center_y in range(max(0, expected_y - search_radius), min(height, expected_y + search_radius + 1)):
        for center_x in range(max(0, expected_x - search_radius), min(width, expected_x + search_radius + 1)):
            ring_y = center_y + ring_y_offsets
            ring_x = center_x + ring_x_offsets
            ring_valid = np.logical_and.reduce((ring_y >= 0, ring_y < height, ring_x >= 0, ring_x < width))
            if not np.any(ring_valid):
                continue
            ring_coverage = float(ring_valid.mean())
            if ring_coverage < 0.85:
                continue
            inner_score = float((mask[ring_y[ring_valid], ring_x[ring_valid]] > 0).mean())

            center_y_samples = center_y + center_y_offsets
            center_x_samples = center_x + center_x_offsets
            center_valid = np.logical_and.reduce(
                (center_y_samples >= 0, center_y_samples < height, center_x_samples >= 0, center_x_samples < width)
            )
            center_leak = (
                float((mask[center_y_samples[center_valid], center_x_samples[center_valid]] > 0).mean())
                if np.any(center_valid)
                else 0.0
            )
            center_error = math.hypot(center_x - expected_center_x, center_y - expected_center_y)
            score = (inner_score * 9.0) - (center_leak * 2.0) - (center_error * 0.18)
            if score > best_score:
                best_score = score
                best = (float(center_x), float(center_y), score)

    if best is None or best[2] < config.inner_ring_center_search_min_score:
        return None
    return best


def _find_inner_ring_template_candidate(
    mask: np.ndarray,
    *,
    search_left: int,
    search_top: int,
    expected_radius: float,
    config: AlignConfig,
) -> tuple[float, float, float, float] | None:
    if mask.size == 0 or np.count_nonzero(mask) == 0:
        return None

    expected_inner_radius = max(4, int(round(expected_radius - config.inner_ring_outer_radius_offset_px)))
    best: tuple[float, float, float, float] | None = None
    best_score = float("-inf")
    half_width = max(1, int(round(config.ring_score_inner_width_px)))
    padding = max(2, int(config.inner_ring_template_padding_px))
    tolerance = max(0, int(config.inner_ring_template_radius_tolerance_px))

    for inner_radius in range(expected_inner_radius - tolerance, expected_inner_radius + tolerance + 1):
        if inner_radius < 4:
            continue
        template, offset_x, offset_y = _inner_ring_template(
            radius=inner_radius,
            half_width=half_width,
            padding=padding,
        )
        if mask.shape[0] < template.shape[0] or mask.shape[1] < template.shape[1]:
            continue
        result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = float(score)
            best = (
                float(max_loc[0] + offset_x + search_left),
                float(max_loc[1] + offset_y + search_top),
                float(inner_radius + config.inner_ring_outer_radius_offset_px),
                float(score),
            )

    return best


def _detect_compass_circle(
    roi: np.ndarray,
    hsv: np.ndarray,
    config: AlignConfig,
    *,
    center_hint_x: float | None = None,
    center_hint_y: float | None = None,
) -> dict[str, float] | None:
    roi_center_x = roi.shape[1] / 2.0
    roi_center_y = roi.shape[0] / 2.0
    expected_radius = _preferred_outer_radius(config)
    if not config.dynamic_center_enabled:
        return {"center_x": roi_center_x, "center_y": roi_center_y, "radius": expected_radius, "score": 0.0}

    strong_ring_mask, weak_ring_mask = _build_compass_ring_masks(roi, hsv, config)
    max_center_error_px = config.circle_detection_max_center_error_px if config.circle_detection_max_center_error_px > 0.0 else None
    search_left, search_top, search_right, search_bottom = _circle_search_bounds(
        width=roi.shape[1],
        height=roi.shape[0],
        center_x=roi_center_x,
        center_y=roi_center_y,
        expected_radius=expected_radius,
        config=config,
        max_center_error_px=max_center_error_px,
    )
    strong_ring_search_mask = strong_ring_mask[search_top:search_bottom, search_left:search_right]
    weak_ring_search_mask = weak_ring_mask[search_top:search_bottom, search_left:search_right]

    best_circle: dict[str, float] | None = None
    best_score = float("-inf")
    candidates: list[dict[str, float]] = [
        {
            "center_x": roi_center_x,
            "center_y": roi_center_y,
            "radius": expected_radius,
            "axis_x": expected_radius,
            "axis_y": expected_radius,
            "angle_degrees": 0.0,
            "source_bonus": 0.0,
        },
    ]
    if center_hint_x is not None and center_hint_y is not None:
        candidates.append(
            {
                "center_x": float(center_hint_x),
                "center_y": float(center_hint_y),
                "radius": expected_radius,
                "axis_x": expected_radius,
                "axis_y": expected_radius,
                "angle_degrees": 0.0,
                "source_bonus": 0.0,
            }
        )

    template_candidate = _find_inner_ring_template_candidate(
        strong_ring_search_mask,
        search_left=search_left,
        search_top=search_top,
        expected_radius=expected_radius,
        config=config,
    )
    if template_candidate is not None:
        template_center_x, template_center_y, template_radius, template_score = template_candidate
        candidates.append(
            {
                "center_x": template_center_x,
                "center_y": template_center_y,
                "radius": template_radius,
                "axis_x": template_radius,
                "axis_y": template_radius,
                "angle_degrees": 0.0,
                "source_bonus": float(template_score * 4.0),
            }
        )

    def add_hough_candidates(
        mask: np.ndarray,
        expected_candidate_radius: float,
        radius_adjustment: float = 0.0,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        if mask.size == 0:
            return
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=max(config.circle_detection_dp, 1.0),
            minDist=max(expected_candidate_radius * 0.75, 12.0),
            param1=max(config.circle_detection_param1, 1.0),
            param2=max(config.circle_detection_param2, 1.0),
            minRadius=max(4, int(round(expected_candidate_radius - config.circle_detection_radius_tolerance_px))),
            maxRadius=max(6, int(round(expected_candidate_radius + config.circle_detection_radius_tolerance_px))),
        )
        if circles is None:
            return
        for circle in circles[0]:
            candidates.append(
                {
                    "center_x": float(circle[0] + offset_x),
                    "center_y": float(circle[1] + offset_y),
                    "radius": float(expected_radius if config.inner_ring_tracking_only else circle[2] + radius_adjustment),
                    "axis_x": float(expected_radius if config.inner_ring_tracking_only else circle[2] + radius_adjustment),
                    "axis_y": float(expected_radius if config.inner_ring_tracking_only else circle[2] + radius_adjustment),
                    "angle_degrees": 0.0,
                    "source_bonus": 0.4,
                }
            )

    if not config.inner_ring_tracking_only:
        add_hough_candidates(
            weak_ring_search_mask,
            expected_radius,
            0.0,
            offset_x=search_left,
            offset_y=search_top,
        )
    add_hough_candidates(
        strong_ring_search_mask,
        max(4.0, expected_radius - config.inner_ring_outer_radius_offset_px),
        0.0 if config.inner_ring_tracking_only else config.inner_ring_outer_radius_offset_px,
        offset_x=search_left,
        offset_y=search_top,
    )

    if not config.inner_ring_tracking_only:
        contours, _ = cv2.findContours(weak_ring_search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < config.ring_min_area:
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            candidates.append(
                {
                    "center_x": float(center_x + search_left),
                    "center_y": float(center_y + search_top),
                    "radius": float(radius),
                    "axis_x": float(radius),
                    "axis_y": float(radius),
                    "angle_degrees": 0.0,
                    "source_bonus": min(0.35, float(area * 0.00005)),
                }
            )

    strong_contours, _ = cv2.findContours(strong_ring_search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in strong_contours:
        area = cv2.contourArea(contour)
        if area < max(12.0, config.ring_min_area * 0.25):
            continue
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center_x = float(ellipse[0][0] + search_left)
            center_y = float(ellipse[0][1] + search_top)
            axis_x = max(1.0, float(ellipse[1][0]) / 2.0)
            axis_y = max(1.0, float(ellipse[1][1]) / 2.0)
            major_axis = max(axis_x, axis_y)
            minor_axis = min(axis_x, axis_y)
            if minor_axis / max(major_axis, 1.0) < config.circle_detection_min_axis_ratio:
                continue
            radius = float(expected_radius if config.inner_ring_tracking_only else (axis_x + axis_y) / 2.0)
            angle_degrees = float(ellipse[2])
        else:
            (raw_center_x, raw_center_y), inner_radius = cv2.minEnclosingCircle(contour)
            center_x = float(raw_center_x + search_left)
            center_y = float(raw_center_y + search_top)
            radius = float(
                expected_radius
                if config.inner_ring_tracking_only
                else inner_radius + config.inner_ring_outer_radius_offset_px
            )
            axis_x = radius
            axis_y = radius
            angle_degrees = 0.0
        candidates.append(
            {
                "center_x": center_x,
                "center_y": center_y,
                "radius": radius,
                "axis_x": axis_x,
                "axis_y": axis_y,
                "angle_degrees": angle_degrees,
                "source_bonus": min(0.5, float(area * 0.00008)),
            }
        )

    deduped_candidates = _dedupe_circle_candidates(candidates)
    scored_candidates: list[tuple[float, dict[str, float]]] = []

    def score_candidate(candidate: dict[str, float]) -> float:
        score = _score_compass_circle_candidate(
            strong_ring_mask=strong_ring_mask,
            weak_ring_mask=weak_ring_mask,
            center_x=candidate["center_x"],
            center_y=candidate["center_y"],
            radius=candidate["radius"],
            axis_x=candidate.get("axis_x"),
            axis_y=candidate.get("axis_y"),
            angle_degrees=candidate.get("angle_degrees", 0.0),
            expected_center_x=roi_center_x,
            expected_center_y=roi_center_y,
            expected_radius=expected_radius,
            max_center_error_px=max_center_error_px,
            config=config,
        )
        return score + candidate.get("source_bonus", 0.0)

    for candidate in deduped_candidates:
        center_x = candidate["center_x"]
        center_y = candidate["center_y"]
        radius = candidate["radius"]
        score = score_candidate(candidate)
        scored_candidates.append((score, candidate))
        if score > best_score:
            best_score = score
            best_circle = {
                "center_x": center_x,
                "center_y": center_y,
                "radius": radius,
                "score": score,
            }

    refinement_candidate: dict[str, float] | None = None
    for score, candidate in sorted(scored_candidates, key=lambda item: item[0], reverse=True):
        if not math.isfinite(score):
            continue
        center_error = math.hypot(candidate["center_x"] - roi_center_x, candidate["center_y"] - roi_center_y)
        if center_error < 6.0:
            continue
        if score < (best_score - 0.75):
            continue
        refinement_candidate = candidate
        break

    if refinement_candidate is not None:
        candidate = refinement_candidate
        refined_center = _find_inner_ring_center_by_offset_search(
            strong_ring_search_mask,
            expected_center_x=candidate["center_x"],
            expected_center_y=candidate["center_y"],
            config=config,
            max_center_error_px=max_center_error_px,
        )
        if refined_center is not None:
            refined_center_x, refined_center_y, _ = refined_center
            refined_candidate = {
                **candidate,
                "center_x": refined_center_x,
                "center_y": refined_center_y,
                "axis_x": candidate["radius"],
                "axis_y": candidate["radius"],
                "angle_degrees": 0.0,
            }
            score = score_candidate(refined_candidate)
            if score > best_score:
                best_score = score
                best_circle = {
                    "center_x": refined_candidate["center_x"],
                    "center_y": refined_candidate["center_y"],
                    "radius": refined_candidate["radius"],
                    "score": score,
                }

    if best_circle is not None and best_score > 0.0:
        template_refined = _refine_compass_circle_with_outer_template(
            roi=roi,
            hsv=hsv,
            center_x=best_circle["center_x"],
            center_y=best_circle["center_y"],
            expected_radius=best_circle["radius"],
            config=config,
            max_center_error_px=max(8.0, config.circle_detection_crop_padding_px),
        )
        if template_refined is not None and (
            template_refined["score"] >= (best_score - 0.35)
            or best_score < 0.9
        ):
            best_score = float(template_refined["score"])
            best_circle = {
                "center_x": float(template_refined["center_x"]),
                "center_y": float(template_refined["center_y"]),
                "radius": float(template_refined["radius"]),
                "score": float(template_refined["score"]),
            }
        elif template_refined is None and center_hint_x is not None and center_hint_y is not None and best_score < 0.9:
            return None

    if (
        best_circle is not None
        and best_score > 0.0
        and center_hint_x is not None
        and center_hint_y is not None
    ):
        hint_distance = math.hypot(best_circle["center_x"] - center_hint_x, best_circle["center_y"] - center_hint_y)
        max_hint_distance = min(
            config.max_marker_distance_px,
            best_circle["radius"] + max(6.0, config.center_dot_circle_center_max_distance_px),
        )
        if hint_distance > max_hint_distance:
            return None

    if best_circle is not None and best_score > 0.0:
        return best_circle

    return None


def _dedupe_circle_candidates(candidates: list[dict[str, float]]) -> list[dict[str, float]]:
    seen: set[tuple[int, int, int]] = set()
    deduped: list[dict[str, float]] = []
    for candidate in candidates:
        if not _is_finite_circle_candidate(candidate):
            continue
        key = (
            int(round(candidate["center_x"] / 3.0)),
            int(round(candidate["center_y"] / 3.0)),
            int(round(candidate["radius"] / 3.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _score_compass_circle_candidate(
    *,
    strong_ring_mask: np.ndarray,
    weak_ring_mask: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    axis_x: float | None,
    axis_y: float | None,
    angle_degrees: float,
    expected_center_x: float,
    expected_center_y: float,
    expected_radius: float,
    max_center_error_px: float | None,
    config: AlignConfig,
) -> float:
    radius_tolerance = (
        max(1.0, float(config.outer_template_radius_tolerance_px))
        if config.outer_template_enabled
        else float(config.circle_detection_radius_tolerance_px)
    )
    radius_error = abs(radius - expected_radius)
    if radius_error > radius_tolerance:
        return float("-inf")

    center_error = math.hypot(center_x - expected_center_x, center_y - expected_center_y)
    if max_center_error_px is not None and center_error > max_center_error_px:
        return float("-inf")

    patch_left, patch_top, patch_right, patch_bottom = _circle_candidate_patch_bounds(
        width=strong_ring_mask.shape[1],
        height=strong_ring_mask.shape[0],
        center_x=center_x,
        center_y=center_y,
        radius=max(radius, expected_radius),
        axis_x=axis_x,
        axis_y=axis_y,
        config=config,
    )
    strong_patch = strong_ring_mask[patch_top:patch_bottom, patch_left:patch_right]
    weak_patch = weak_ring_mask[patch_top:patch_bottom, patch_left:patch_right]
    local_center_x = center_x - patch_left
    local_center_y = center_y - patch_top
    yy, xx = np.ogrid[: strong_patch.shape[0], : strong_patch.shape[1]]
    dx = xx - local_center_x
    dy = yy - local_center_y
    distances = np.sqrt((dx**2) + (dy**2))
    angles = (np.arctan2(dy, dx) + (2.0 * np.pi)) % (2.0 * np.pi)
    effective_axis_x = max(1.0, float(axis_x if axis_x is not None else radius))
    effective_axis_y = max(1.0, float(axis_y if axis_y is not None else radius))
    ellipse_distortion = abs(effective_axis_x - effective_axis_y)
    inner_width = config.ring_score_inner_width_px + min(4.0, ellipse_distortion * 0.35)
    outer_width = config.ring_score_outer_width_px + min(4.0, ellipse_distortion * 0.35)

    inner_radius = (
        max(1.0, config.inner_ring_radius_px)
        if config.inner_ring_tracking_only
        else max(1.0, radius - config.inner_ring_outer_radius_offset_px)
    )
    inner_radius_for_fill = max(1.0, inner_radius - 3.0 - (ellipse_distortion * 0.2))
    inner_annulus = np.logical_and(
        distances >= (inner_radius - inner_width),
        distances <= (inner_radius + inner_width),
    )
    center_disk = distances <= inner_radius_for_fill

    inner_score = float((strong_patch[inner_annulus] > 0).mean()) if np.any(inner_annulus) else 0.0
    center_leak = float((weak_patch[center_disk] > 0).mean()) if np.any(center_disk) else 0.0

    angular_bins = max(4, config.ring_score_angular_bins)
    angular_hits = 0
    ring_bin_hits: list[bool] = []
    if np.any(inner_annulus):
        inner_pixels = strong_patch[inner_annulus] > 0
        inner_angles = angles[inner_annulus]
        for bin_index in range(angular_bins):
            start_angle = (2.0 * np.pi * bin_index) / angular_bins
            end_angle = (2.0 * np.pi * (bin_index + 1)) / angular_bins
            bin_mask = np.logical_and(inner_angles >= start_angle, inner_angles < end_angle)
            hit = np.any(bin_mask) and float(inner_pixels[bin_mask].mean()) >= 0.08
            ring_bin_hits.append(hit)
            if hit:
                angular_hits += 1
    angular_coverage = angular_hits / angular_bins
    if config.inner_ring_tracking_only:
        if inner_score < 0.18 or angular_coverage < 0.35:
            return float("-inf")
        return (
            (inner_score * 9.0)
            + (angular_coverage * 3.0)
            - (radius_error * 0.05)
            - (center_error * 0.08)
            - (center_leak * 2.0)
        )

    outer_annulus = np.logical_and(
        distances >= (radius - outer_width),
        distances <= (radius + outer_width),
    )
    outer_score = float((weak_patch[outer_annulus] > 0).mean()) if np.any(outer_annulus) else 0.0
    outer_bin_hits: list[bool] = []
    if np.any(outer_annulus):
        outer_pixels = weak_patch[outer_annulus] > 0
        outer_angles = angles[outer_annulus]
        for bin_index in range(angular_bins):
            start_angle = (2.0 * np.pi * bin_index) / angular_bins
            end_angle = (2.0 * np.pi * (bin_index + 1)) / angular_bins
            bin_mask = np.logical_and(outer_angles >= start_angle, outer_angles < end_angle)
            hit = np.any(bin_mask) and float(outer_pixels[bin_mask].mean()) >= 0.08
            outer_bin_hits.append(hit)
    segment_groups = _circular_true_group_count(outer_bin_hits)
    segment_score = max(0.0, 1.0 - (abs(segment_groups - 4) / 4.0))

    if inner_score < 0.10:
        return float("-inf")
    if outer_score < 0.03 and angular_coverage < 0.20:
        return float("-inf")

    return (
        (inner_score * 7.0)
        + (outer_score * 2.0)
        + (angular_coverage * 1.5)
        + (segment_score * 1.5)
        - (radius_error * 0.35)
        - (center_error * 0.10)
        - (center_leak * 1.5)
    )


def _circle_candidate_patch_bounds(
    *,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    radius: float,
    axis_x: float | None,
    axis_y: float | None,
    config: AlignConfig,
) -> tuple[int, int, int, int]:
    span = max(radius, axis_x or radius, axis_y or radius) + config.ring_score_candidate_padding_px
    left = max(0, int(math.floor(center_x - span)))
    top = max(0, int(math.floor(center_y - span)))
    right = min(width, int(math.ceil(center_x + span + 1.0)))
    bottom = min(height, int(math.ceil(center_y + span + 1.0)))
    return left, top, right, bottom


def _circular_true_group_count(values: list[bool]) -> int:
    if not values or not any(values):
        return 0
    groups = 0
    previous = values[-1]
    for current in values:
        if current and not previous:
            groups += 1
        previous = current
    return groups


def _candidate_components(
    mask: np.ndarray,
    hsv: np.ndarray,
    config: AlignConfig,
    *,
    local_center_x: float | None = None,
    local_center_y: float | None = None,
) -> list[dict[str, float]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if local_center_x is None:
        local_center_x = config.roi_size / 2.0
    if local_center_y is None:
        local_center_y = config.roi_size / 2.0
    candidates: list[dict[str, float]] = []

    for index in range(1, num_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < config.marker_min_area or area > config.marker_max_area:
            continue

        centroid_x = float(centroids[index][0])
        centroid_y = float(centroids[index][1])
        if math.hypot(centroid_x - local_center_x, centroid_y - local_center_y) > config.max_marker_distance_px:
            continue

        component_mask = labels == index
        mean_value = float(hsv[:, :, 2][component_mask].mean())
        candidates.append(
            {
                "area": float(area),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "mean_value": mean_value,
            }
        )

    candidates.sort(key=lambda candidate: (candidate["area"], candidate["mean_value"]), reverse=True)
    return candidates


def _occupancy_scores(
    mask: np.ndarray,
    centroid_x: float,
    centroid_y: float,
    config: AlignConfig,
) -> tuple[float, float]:
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    squared_distance = (xx - centroid_x) ** 2 + (yy - centroid_y) ** 2

    inner_mask = squared_distance <= config.inner_disk_radius_px**2
    outer_mask = squared_distance <= config.outer_disk_radius_px**2
    outer_ring_mask = np.logical_and(outer_mask, np.logical_not(inner_mask))

    inner_pixels = mask[inner_mask] > 0
    outer_ring_pixels = mask[outer_ring_mask] > 0

    inner_occupancy = float(inner_pixels.mean()) if inner_pixels.size else 0.0
    outer_ring_occupancy = float(outer_ring_pixels.mean()) if outer_ring_pixels.size else 0.0
    return inner_occupancy, outer_ring_occupancy


def _refine_hollow_center(
    mask: np.ndarray,
    centroid_x: float,
    centroid_y: float,
    config: AlignConfig,
) -> dict[str, float] | None:
    best_candidate: dict[str, float] | None = None
    search_radius = max(config.refinement_search_radius_px, config.outer_disk_radius_px)
    for offset_x in range(-search_radius, search_radius + 1):
        for offset_y in range(-search_radius, search_radius + 1):
            candidate_x = centroid_x + offset_x
            candidate_y = centroid_y + offset_y
            inner_occupancy, outer_ring_occupancy = _occupancy_scores(mask, candidate_x, candidate_y, config)
            if inner_occupancy >= config.filled_inner_occupancy_threshold or outer_ring_occupancy <= 0.0:
                continue

            displacement = math.hypot(offset_x, offset_y)
            score = outer_ring_occupancy - inner_occupancy - (displacement * 0.06)
            candidate = {
                "score": score,
                "x": candidate_x,
                "y": candidate_y,
                "inner_occupancy": inner_occupancy,
                "outer_ring_occupancy": outer_ring_occupancy,
            }
            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

    return best_candidate


def _extract_roi(image: Any, region: Region) -> np.ndarray:
    x, y, width, height = region
    if image.shape[1] < x + width or image.shape[0] < y + height:
        raise ValueError("Image does not match the calibrated ROI bounds for compass detection.")
    return image[y : y + height, x : x + width]


def _rotate_offset(dx: float, dy: float, rotation_radians: float) -> tuple[float, float]:
    cosine = math.cos(rotation_radians)
    sine = math.sin(rotation_radians)
    return (cosine * dx) - (sine * dy), (sine * dx) + (cosine * dy)
