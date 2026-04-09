from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
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


WINDOW_TITLE = "Elite Dangerous"


@dataclass(slots=True)
class ReticleTrackerConfig:
    search_radius_px: int = 500
    expected_outer_radius_px: float = 45.0
    expected_inner_radius_px: float = 46.0
    template_radius_candidates_px: tuple[int, ...] = (44, 45, 46)
    template_padding_px: int = 18
    template_coarse_scale: float = 0.5
    template_refine_margin_px: int = 12
    hud_hue_center: int = 107
    hud_hue_tolerance: int = 12
    hud_saturation_threshold: int = 70
    hud_value_threshold: int = 55
    prototype_bgr: tuple[float, float, float] = (206.0, 162.0, 51.0)
    prototype_distance_threshold: float = 58.0
    prototype_distance_fallback_threshold: float = 70.0
    mask_dilate_iterations: int = 1
    radius_tolerance_px: float = 14.0
    annulus_half_width_px: float = 2.0
    tick_width_px: int = 4
    tick_length_px: int = 16
    hough_dp: float = 1.2
    hough_param1: float = 80.0
    hough_param2: float = 12.0
    min_score: float = 1.10
    max_center_distance_px: float = 1_000.0
    max_missing_sector_occ: float = 0.90
    max_inner_leak: float = 0.18
    min_tick_occ: float = 0.22
    min_visible_sector_occ: float = 0.22
    min_visible_sector_balance: float = 0.45
    angular_bin_count: int = 72
    min_visible_arc_fraction: float = 0.55
    min_visible_arc_run_fraction: float = 0.38
    candidate_padding_px: int = 24
    debug_scale: int = 2
    capture_target_fps: int = 30
    show_mask_window: bool = True


@dataclass(slots=True)
class ReticleDetection:
    found: bool
    center_x: float | None = None
    center_y: float | None = None
    outer_radius_px: float | None = None
    score: float | None = None
    search_region: tuple[int, int, int, int] | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def detect_center_reticle(image: np.ndarray, config: ReticleTrackerConfig) -> tuple[ReticleDetection, np.ndarray, np.ndarray]:
    if image is None:
        return ReticleDetection(found=False), np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)

    crop, search_region = _extract_center_crop(image, config)
    mask = _build_reticle_mask(crop, config)
    best = _find_best_reticle_candidate(mask, config)
    overlay = _build_debug_overlay(crop, mask, best, config)

    if best is None:
        return ReticleDetection(found=False, search_region=search_region), overlay, mask

    center_x = search_region[0] + best["center_x"]
    center_y = search_region[1] + best["center_y"]
    detection = ReticleDetection(
        found=True,
        center_x=center_x,
        center_y=center_y,
        outer_radius_px=best["radius"],
        score=best["score"],
        search_region=search_region,
        metrics=best,
    )
    return detection, overlay, mask


def _extract_center_crop(image: np.ndarray, config: ReticleTrackerConfig) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    half_width = max(32, int(config.search_radius_px))
    half_height = max(32, int(config.search_radius_px))
    left = max(0, center_x - half_width)
    top = max(0, center_y - half_height)
    right = min(width, center_x + half_width)
    bottom = min(height, center_y + half_height)
    crop = image[top:bottom, left:right].copy()
    return crop, (left, top, right - left, bottom - top)


def _build_reticle_mask(crop: np.ndarray, config: ReticleTrackerConfig) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue_distance = np.minimum(np.abs(hue - config.hud_hue_center), 180 - np.abs(hue - config.hud_hue_center))
    hsv_mask = np.where(
        (hue_distance <= config.hud_hue_tolerance)
        & (sat >= config.hud_saturation_threshold)
        & (val >= config.hud_value_threshold),
        255,
        0,
    ).astype(np.uint8)

    prototype = np.array(config.prototype_bgr, dtype=np.float32)
    diff = crop.astype(np.float32) - prototype
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    prototype_mask = np.where(distance <= config.prototype_distance_threshold, 255, 0).astype(np.uint8)
    if np.count_nonzero(prototype_mask) < 24:
        prototype_mask = np.where(distance <= config.prototype_distance_fallback_threshold, 255, 0).astype(np.uint8)

    mask = cv2.bitwise_or(hsv_mask, prototype_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    if config.mask_dilate_iterations > 0:
        mask = cv2.dilate(mask, kernel, iterations=config.mask_dilate_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _find_best_reticle_candidate(mask: np.ndarray, config: ReticleTrackerConfig) -> dict[str, float] | None:
    template_best = _find_template_reticle_candidate(mask, config)

    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=max(config.hough_dp, 1.0),
        minDist=max(config.expected_outer_radius_px * 0.6, 40.0),
        param1=max(config.hough_param1, 1.0),
        param2=max(config.hough_param2, 1.0),
        minRadius=max(12, int(round(config.expected_outer_radius_px - config.radius_tolerance_px))),
        maxRadius=max(16, int(round(config.expected_outer_radius_px + config.radius_tolerance_px))),
    )

    candidates: list[tuple[float, float, float]] = []
    if circles is not None:
        for circle in circles[0]:
            candidates.append((float(circle[0]), float(circle[1]), float(circle[2])))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 24.0:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if abs(float(radius) - config.expected_outer_radius_px) > (config.radius_tolerance_px + 4.0):
            continue
        candidates.append((float(center_x), float(center_y), float(radius)))

    if not candidates:
        if template_best is None or not _is_viable_reticle_candidate(template_best, config):
            return None
        return template_best

    candidates = _dedupe_candidates(candidates)
    center_x = mask.shape[1] / 2.0
    center_y = mask.shape[0] / 2.0
    best: dict[str, float] | None = None
    best_score = float("-inf")
    if template_best is not None and _is_viable_reticle_candidate(template_best, config):
        best = template_best
        best_score = template_best["score"]
    for cand_x, cand_y, cand_radius in candidates:
        metrics = _score_reticle_candidate(
            mask=mask,
            center_x=cand_x,
            center_y=cand_y,
            radius=cand_radius,
            expected_center_x=center_x,
            expected_center_y=center_y,
            config=config,
        )
        if not _is_viable_reticle_candidate(metrics, config):
            continue
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best = metrics

    if best is None or best["score"] < config.min_score:
        return None
    return best


def _find_template_reticle_candidate(mask: np.ndarray, config: ReticleTrackerConfig) -> dict[str, float] | None:
    expected_center_x = mask.shape[1] / 2.0
    expected_center_y = mask.shape[0] / 2.0
    best: dict[str, float] | None = None
    best_score = float("-inf")
    coarse_scale = min(1.0, max(0.25, float(config.template_coarse_scale)))
    coarse_mask = (
        cv2.resize(mask, None, fx=coarse_scale, fy=coarse_scale, interpolation=cv2.INTER_NEAREST)
        if coarse_scale < 0.999
        else mask
    )

    for radius in config.template_radius_candidates_px:
        template, center_offset_x, center_offset_y = _reticle_template(
            radius=radius,
            inner_radius=max(1, radius - 2),
            tick_width=config.tick_width_px,
            tick_length=config.tick_length_px,
            padding=config.template_padding_px,
        )
        if mask.shape[0] < template.shape[0] or mask.shape[1] < template.shape[1]:
            continue
        coarse_template = (
            cv2.resize(template, None, fx=coarse_scale, fy=coarse_scale, interpolation=cv2.INTER_NEAREST)
            if coarse_scale < 0.999
            else template
        )
        if coarse_mask.shape[0] < coarse_template.shape[0] or coarse_mask.shape[1] < coarse_template.shape[1]:
            continue
        coarse_result = cv2.matchTemplate(coarse_mask, coarse_template, cv2.TM_CCOEFF_NORMED)
        _, coarse_score, _, coarse_loc = cv2.minMaxLoc(coarse_result)
        coarse_x = int(round(coarse_loc[0] / coarse_scale)) if coarse_scale < 0.999 else coarse_loc[0]
        coarse_y = int(round(coarse_loc[1] / coarse_scale)) if coarse_scale < 0.999 else coarse_loc[1]
        refine_margin = max(4, int(config.template_refine_margin_px))
        search_left = max(0, coarse_x - refine_margin)
        search_top = max(0, coarse_y - refine_margin)
        search_right = min(mask.shape[1], coarse_x + template.shape[1] + refine_margin)
        search_bottom = min(mask.shape[0], coarse_y + template.shape[0] + refine_margin)
        search_region = mask[search_top:search_bottom, search_left:search_right]
        if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
            continue
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, refine_score, _, refine_loc = cv2.minMaxLoc(result)
        template_score = max(float(coarse_score), float(refine_score))
        max_loc = (search_left + refine_loc[0], search_top + refine_loc[1])
        center_x = float(max_loc[0] + center_offset_x)
        center_y = float(max_loc[1] + center_offset_y)
        metrics = _score_template_candidate(
            mask=mask,
            template=template,
            template_top_left=max_loc,
            center_x=center_x,
            center_y=center_y,
            radius=float(radius),
            template_score=float(template_score),
            expected_center_x=expected_center_x,
            expected_center_y=expected_center_y,
            config=config,
        )
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best = metrics

    return best


def _score_template_candidate(
    *,
    mask: np.ndarray,
    template: np.ndarray,
    template_top_left: tuple[int, int],
    center_x: float,
    center_y: float,
    radius: float,
    template_score: float,
    expected_center_x: float,
    expected_center_y: float,
    config: ReticleTrackerConfig,
) -> dict[str, float]:
    top_left_x, top_left_y = template_top_left
    template_region = mask[top_left_y : top_left_y + template.shape[0], top_left_x : top_left_x + template.shape[1]]
    template_on = template > 0
    overlap = float((template_region[template_on] > 0).mean()) if np.any(template_on) else 0.0

    center_distance = math.hypot(center_x - expected_center_x, center_y - expected_center_y)
    metrics = _score_reticle_candidate(
        mask=mask,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        expected_center_x=expected_center_x,
        expected_center_y=expected_center_y,
        config=config,
    )
    geometry_score = metrics["score"]
    score = (
        geometry_score
        + (template_score * 2.2)
        + (overlap * 1.6)
        - (metrics["inner_leak"] * 0.5)
        - (center_distance * 0.0002)
    )
    metrics["template_score"] = template_score
    metrics["template_overlap"] = overlap
    metrics["geometry_score"] = geometry_score
    metrics["score"] = score
    return metrics


def _dedupe_candidates(candidates: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    seen: set[tuple[int, int, int]] = set()
    deduped: list[tuple[float, float, float]] = []
    for center_x, center_y, radius in candidates:
        key = (int(round(center_x / 3.0)), int(round(center_y / 3.0)), int(round(radius / 3.0)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((center_x, center_y, radius))
    return deduped


def _score_reticle_candidate(
    *,
    mask: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    expected_center_x: float,
    expected_center_y: float,
    config: ReticleTrackerConfig,
) -> dict[str, float]:
    patch, patch_left, patch_top = _extract_candidate_patch(mask, center_x, center_y, radius, config)
    local_center_x = center_x - patch_left
    local_center_y = center_y - patch_top
    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    distances = np.sqrt(((xx - local_center_x) ** 2) + ((yy - local_center_y) ** 2))
    angles = (np.degrees(np.arctan2(yy - local_center_y, xx - local_center_x)) + 360.0) % 360.0
    ring_mask = np.logical_and(
        distances >= (radius - config.annulus_half_width_px),
        distances <= (radius + config.annulus_half_width_px),
    )
    inner_disk = distances <= max(1.0, radius - 8.0)

    center_distance = math.hypot(center_x - expected_center_x, center_y - expected_center_y)
    radius_error = abs(radius - config.expected_outer_radius_px)
    total_ring_occ = float((patch[ring_mask] > 0).mean()) if np.any(ring_mask) else 0.0
    inner_leak = float((patch[inner_disk] > 0).mean()) if np.any(inner_disk) else 0.0

    sector_defs = {
        "missing_tr": lambda a: np.logical_or(a >= 300.0, a < 30.0),
        "left_top": lambda a: np.logical_and(a >= 30.0, a < 120.0),
        "left_bottom": lambda a: np.logical_and(a >= 120.0, a < 210.0),
        "right_bottom": lambda a: np.logical_and(a >= 210.0, a < 300.0),
    }
    sector_occ: dict[str, float] = {}
    for name, selector in sector_defs.items():
        sector_mask = np.logical_and(ring_mask, selector(angles))
        sector_occ[name] = float((patch[sector_mask] > 0).mean()) if np.any(sector_mask) else 0.0

    visible_sector_values = (
        sector_occ["left_top"],
        sector_occ["left_bottom"],
        sector_occ["right_bottom"],
    )
    visible_sector_min = min(visible_sector_values)
    visible_sector_max = max(visible_sector_values)
    visible_sector_balance = (
        visible_sector_min / visible_sector_max
        if visible_sector_max > 1e-6
        else 0.0
    )

    visible_arc_fraction, visible_arc_run_fraction = _measure_visible_arc_support(
        mask=patch,
        ring_mask=ring_mask,
        angles=angles,
        config=config,
    )

    tick_mask = (
        (np.abs(xx - local_center_x) <= config.tick_width_px)
        & (yy >= local_center_y + radius - 2.0)
        & (yy <= local_center_y + radius + config.tick_length_px)
    )
    tick_occ = float((patch[tick_mask] > 0).mean()) if np.any(tick_mask) else 0.0

    score = (
        (sector_occ["left_top"] * 2.4)
        + (sector_occ["left_bottom"] * 2.9)
        + (sector_occ["right_bottom"] * 1.8)
        + (tick_occ * 1.8)
        + (total_ring_occ * 1.2)
        + (visible_sector_balance * 1.4)
        + (visible_arc_fraction * 1.8)
        + (visible_arc_run_fraction * 1.6)
        - (sector_occ["missing_tr"] * 2.9)
        - (inner_leak * 2.4)
        - (radius_error * 0.10)
        - (center_distance * 0.0015)
    )

    return {
        "center_x": center_x,
        "center_y": center_y,
        "radius": radius,
        "score": score,
        "center_distance": center_distance,
        "radius_error": radius_error,
        "total_ring_occ": total_ring_occ,
        "missing_tr_occ": sector_occ["missing_tr"],
        "left_top_occ": sector_occ["left_top"],
        "left_bottom_occ": sector_occ["left_bottom"],
        "right_bottom_occ": sector_occ["right_bottom"],
        "visible_sector_min": visible_sector_min,
        "visible_sector_balance": visible_sector_balance,
        "visible_arc_fraction": visible_arc_fraction,
        "visible_arc_run_fraction": visible_arc_run_fraction,
        "tick_occ": tick_occ,
        "inner_leak": inner_leak,
    }


def _extract_candidate_patch(
    mask: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    config: ReticleTrackerConfig,
) -> tuple[np.ndarray, int, int]:
    span = int(math.ceil(radius + config.tick_length_px + config.candidate_padding_px))
    left = max(0, int(math.floor(center_x - span)))
    top = max(0, int(math.floor(center_y - span)))
    right = min(mask.shape[1], int(math.ceil(center_x + span + 1.0)))
    bottom = min(mask.shape[0], int(math.ceil(center_y + span + 1.0)))
    return mask[top:bottom, left:right], left, top


def _measure_visible_arc_support(
    *,
    mask: np.ndarray,
    ring_mask: np.ndarray,
    angles: np.ndarray,
    config: ReticleTrackerConfig,
) -> tuple[float, float]:
    bin_count = max(12, int(config.angular_bin_count))
    ring_support = np.logical_and(ring_mask, mask > 0)
    bin_has_support = np.zeros(bin_count, dtype=np.bool_)
    if np.any(ring_support):
        ring_angles = angles[ring_support]
        bin_indices = np.floor((ring_angles / 360.0) * bin_count).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, bin_count - 1)
        bin_has_support[np.unique(bin_indices)] = True

    bin_centers = (np.arange(bin_count, dtype=np.float32) + 0.5) * (360.0 / bin_count)
    visible_bins = np.logical_and(bin_centers >= 30.0, bin_centers < 300.0)
    visible_support = bin_has_support[visible_bins]
    if visible_support.size == 0:
        return 0.0, 0.0
    visible_arc_fraction = float(visible_support.mean())
    visible_arc_run_fraction = _longest_true_run_fraction(visible_support)
    return visible_arc_fraction, visible_arc_run_fraction


def _longest_true_run_fraction(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    longest = 0
    run = 0
    for value in values:
        if bool(value):
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return float(longest / values.size)


def _is_viable_reticle_candidate(metrics: dict[str, float], config: ReticleTrackerConfig) -> bool:
    if metrics["score"] < config.min_score:
        return False
    if metrics["center_distance"] > config.max_center_distance_px:
        return False
    if metrics["missing_tr_occ"] > config.max_missing_sector_occ:
        return False
    if metrics["inner_leak"] > config.max_inner_leak:
        return False
    if metrics["tick_occ"] < config.min_tick_occ:
        return False
    if metrics["visible_sector_min"] < config.min_visible_sector_occ:
        return False
    if metrics["visible_sector_balance"] < config.min_visible_sector_balance:
        return False
    if metrics["visible_arc_fraction"] < config.min_visible_arc_fraction:
        return False
    if metrics["visible_arc_run_fraction"] < config.min_visible_arc_run_fraction:
        return False
    return True


@lru_cache(maxsize=16)
def _reticle_template(
    *,
    radius: int,
    inner_radius: int,
    tick_width: int,
    tick_length: int,
    padding: int,
) -> tuple[np.ndarray, int, int]:
    width = int((radius * 2) + (padding * 2))
    height = int((radius * 2) + (padding * 2) + tick_length + padding)
    center_x = padding + radius
    center_y = padding + radius
    yy, xx = np.ogrid[:height, :width]
    distances = np.sqrt(((xx - center_x) ** 2) + ((yy - center_y) ** 2))
    angles = (np.degrees(np.arctan2(yy - center_y, xx - center_x)) + 360.0) % 360.0
    ring = np.logical_and(distances >= inner_radius, distances <= radius)
    missing_sector = np.logical_or(angles >= 300.0, angles < 30.0)
    tick = (
        (np.abs(xx - center_x) <= tick_width)
        & (yy >= center_y + radius - 2.0)
        & (yy <= center_y + radius + tick_length)
    )
    template = np.where(np.logical_or(np.logical_and(ring, ~missing_sector), tick), 255, 0).astype(np.uint8)
    return template, center_x, center_y


def _build_debug_overlay(
    crop: np.ndarray,
    mask: np.ndarray,
    best: dict[str, float] | None,
    config: ReticleTrackerConfig,
) -> np.ndarray:
    overlay = crop.copy()
    center = (overlay.shape[1] // 2, overlay.shape[0] // 2)
    cv2.drawMarker(overlay, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)
    expected_radius = int(round(config.expected_outer_radius_px))
    cv2.circle(overlay, center, expected_radius, (0, 160, 255), 1)
    cv2.circle(overlay, center, max(1, expected_radius - 2), (255, 220, 0), 1)
    cv2.rectangle(
        overlay,
        (center[0] - config.tick_width_px, center[1] + expected_radius - 2),
        (center[0] + config.tick_width_px, center[1] + expected_radius + config.tick_length_px),
        (0, 120, 255),
        1,
    )

    if best is not None:
        circle_center = (int(round(best["center_x"])), int(round(best["center_y"])))
        radius = int(round(best["radius"]))
        cv2.circle(overlay, circle_center, radius, (0, 255, 255), 2)
        cv2.circle(overlay, circle_center, max(1, radius - 2), (255, 200, 0), 1)
        cv2.line(overlay, center, circle_center, (0, 255, 0), 1)
        cv2.rectangle(
            overlay,
            (circle_center[0] - config.tick_width_px, circle_center[1] + radius - 2),
            (circle_center[0] + config.tick_width_px, circle_center[1] + radius + config.tick_length_px),
            (0, 200, 255),
            1,
        )

    scale = max(1, int(config.debug_scale))
    overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    lines = [
        f"expected_radius={config.expected_outer_radius_px:.1f}",
        f"expected_center=({center[0]},{center[1]})",
        "found=no" if best is None else (
            f"found=yes score={best['score']:.2f} center=({best['center_x']:.1f},{best['center_y']:.1f}) r={best['radius']:.1f}"
        ),
    ]
    if best is not None:
        lines.extend(
            [
                f"occ lt={best['left_top_occ']:.2f} lb={best['left_bottom_occ']:.2f} rb={best['right_bottom_occ']:.2f}",
                f"missing_tr={best['missing_tr_occ']:.2f} tick={best['tick_occ']:.2f} leak={best['inner_leak']:.2f}",
                f"arc vis={best['visible_arc_fraction']:.2f} run={best['visible_arc_run_fraction']:.2f} bal={best['visible_sector_balance']:.2f}",
                f"center_err={best['center_distance']:.1f} radius_err={best['radius_error']:.1f}",
            ]
        )
    y = 20
    for line in lines:
        cv2.putText(overlay, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return overlay


def _show_image_debug(image: np.ndarray, title_prefix: str, config: ReticleTrackerConfig) -> None:
    detection, overlay, mask = detect_center_reticle(image, config)
    print(
        {
            "title": title_prefix,
            "found": detection.found,
            "center_x": None if detection.center_x is None else round(detection.center_x, 2),
            "center_y": None if detection.center_y is None else round(detection.center_y, 2),
            "outer_radius_px": None if detection.outer_radius_px is None else round(detection.outer_radius_px, 2),
            "score": None if detection.score is None else round(detection.score, 3),
            "metrics": {k: round(v, 3) for k, v in detection.metrics.items()},
        }
    )
    cv2.imshow("Reticle Tracker Debug", overlay)
    if config.show_mask_window:
        cv2.imshow(
            "Reticle Tracker Mask",
            cv2.resize(mask, None, fx=max(1, config.debug_scale), fy=max(1, config.debug_scale), interpolation=cv2.INTER_NEAREST),
        )


def _run_live(config: ReticleTrackerConfig) -> int:
    context = build_standalone_context(AppConfig.default())
    capture = context.capture
    if capture is None:
        raise RuntimeError("Capture is not available.")

    window_capture_region = _find_window_client_region(WINDOW_TITLE)
    if window_capture_region is None:
        raise RuntimeError(f"Could not locate the {WINDOW_TITLE!r} window.")
    output_index, capture_region = _resolve_output_region(window_capture_region)
    if output_index is None or capture_region is None:
        raise RuntimeError("Could not map the Elite window onto a dxcam output.")
    if isinstance(capture, DxcamCapture):
        capture.set_output_index(output_index)
        previous_fps = capture.set_target_fps(config.capture_target_fps)
    else:
        previous_fps = None

    print(f"Tracking live reticle on output {output_index}, region={capture_region}. Press q to quit.")
    try:
        while True:
            frame = capture.grab(region=capture_region)
            if frame is None:
                continue
            _show_image_debug(frame, "live", config)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
    finally:
        if isinstance(capture, DxcamCapture):
            capture.set_target_fps(previous_fps)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return 0


def _run_images(image_paths: list[str], config: ReticleTrackerConfig) -> int:
    for path_str in image_paths:
        path = Path(path_str)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not load image: {path}")
        _show_image_debug(image, path.name, config)
        key = cv2.waitKey(0) & 0xFF
        if key in {ord("q"), 27}:
            break
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug tracker for the center-screen final alignment reticle.")
    parser.add_argument("images", nargs="*", help="Optional image paths to inspect instead of live capture.")
    parser.add_argument("--no-mask-window", action="store_true", help="Hide the binary mask debug window.")
    parser.add_argument("--search-radius", type=int, default=260)
    parser.add_argument("--debug-scale", type=int, default=2)
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = ReticleTrackerConfig(
        search_radius_px=args.search_radius,
        debug_scale=args.debug_scale,
        show_mask_window=not args.no_mask_window,
    )
    if args.images:
        return _run_images(args.images, config)
    return _run_live(config)


if __name__ == "__main__":
    raise SystemExit(_main())
