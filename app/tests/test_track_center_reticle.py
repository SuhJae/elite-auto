from __future__ import annotations

import unittest

import numpy as np

from app.actions.track_center_reticle import ReticleTrackerConfig, detect_center_reticle


def make_reticle_image(
    *,
    width: int = 1280,
    height: int = 720,
    center_x: int,
    center_y: int,
    radius: float = 41.0,
    color: tuple[int, int, int] = (220, 170, 40),
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:height, :width]
    distances = np.sqrt(((xx - center_x) ** 2) + ((yy - center_y) ** 2))
    angles = (np.degrees(np.arctan2(yy - center_y, xx - center_x)) + 360.0) % 360.0
    ring = np.logical_and(distances >= (radius - 1.2), distances <= (radius + 1.2))
    missing_sector = np.logical_or(angles >= 300.0, angles < 30.0)
    image[np.logical_and(ring, ~missing_sector)] = color
    tick = (
        (np.abs(xx - center_x) <= 2)
        & (yy >= center_y + radius - 2.0)
        & (yy <= center_y + radius + 16.0)
    )
    image[tick] = color
    return image


def add_full_circle(
    image: np.ndarray,
    *,
    center_x: int,
    center_y: int,
    radius: float,
    color: tuple[int, int, int] = (220, 170, 40),
) -> None:
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    distances = np.sqrt(((xx - center_x) ** 2) + ((yy - center_y) ** 2))
    ring = np.logical_and(distances >= (radius - 1.2), distances <= (radius + 1.2))
    image[ring] = color


def add_arc_fragment_with_hud_clutter(
    image: np.ndarray,
    *,
    center_x: int,
    center_y: int,
    radius: float,
    color: tuple[int, int, int] = (220, 170, 40),
) -> None:
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    distances = np.sqrt(((xx - center_x) ** 2) + ((yy - center_y) ** 2))
    angles = (np.degrees(np.arctan2(yy - center_y, xx - center_x)) + 360.0) % 360.0
    ring = np.logical_and(distances >= (radius - 1.2), distances <= (radius + 1.2))
    arc = np.logical_and(ring, np.logical_and(angles >= 110.0, angles <= 250.0))
    image[arc] = color
    for top in range(center_y - 37, center_y + 43, 18):
        image[top : top + 4, center_x + 54 : center_x + 129] = color
        image[top + 8 : top + 12, center_x + 64 : center_x + 114 : 8] = color
    image[center_y + 223 : center_y + 233, center_x + 94 : center_x + 294] = color


def add_label_clutter(
    image: np.ndarray,
    *,
    anchor_x: int,
    anchor_y: int,
    color: tuple[int, int, int] = (255, 180, 0),
) -> None:
    for top in range(anchor_y - 20, anchor_y + 30, 18):
        image[top : top + 4, anchor_x : anchor_x + 82] = color
        image[top + 8 : top + 12, anchor_x + 12 : anchor_x + 76 : 8] = color


class TestTrackCenterReticle(unittest.TestCase):
    def test_detects_partial_reticle_near_screen_center(self) -> None:
        config = ReticleTrackerConfig(search_radius_px=260)
        image = make_reticle_image(center_x=690, center_y=330)

        detection, _, _ = detect_center_reticle(image, config)

        self.assertTrue(detection.found)
        assert detection.center_x is not None
        assert detection.center_y is not None
        assert detection.outer_radius_px is not None
        self.assertLess(abs(detection.center_x - 690), 4.0)
        self.assertLess(abs(detection.center_y - 330), 4.0)
        self.assertLess(abs(detection.outer_radius_px - config.expected_outer_radius_px), 6.0)

    def test_prefers_missing_quarter_reticle_over_full_circle_distractor(self) -> None:
        config = ReticleTrackerConfig(search_radius_px=260)
        image = make_reticle_image(center_x=680, center_y=350)
        add_full_circle(image, center_x=600, center_y=360, radius=config.expected_outer_radius_px)

        detection, _, _ = detect_center_reticle(image, config)

        self.assertTrue(detection.found)
        assert detection.center_x is not None
        assert detection.center_y is not None
        self.assertLess(abs(detection.center_x - 680), 6.0)
        self.assertLess(abs(detection.center_y - 350), 6.0)

    def test_detects_elite_blue_reticle_near_screen_center(self) -> None:
        config = ReticleTrackerConfig(search_radius_px=260)
        image = make_reticle_image(center_x=690, center_y=330, color=(255, 180, 0))

        detection, _, _ = detect_center_reticle(image, config)

        self.assertTrue(detection.found)
        assert detection.center_x is not None
        assert detection.center_y is not None
        self.assertLess(abs(detection.center_x - 690), 4.0)
        self.assertLess(abs(detection.center_y - 330), 4.0)

    def test_detects_off_center_reticle_for_steering(self) -> None:
        config = ReticleTrackerConfig(search_radius_px=260)
        image = make_reticle_image(center_x=840, center_y=210, color=(255, 180, 0))
        add_label_clutter(image, anchor_x=875, anchor_y=205, color=(255, 180, 0))

        detection, _, _ = detect_center_reticle(image, config)

        self.assertTrue(detection.found)
        assert detection.center_x is not None
        assert detection.center_y is not None
        self.assertLess(abs(detection.center_x - 840), 8.0)
        self.assertLess(abs(detection.center_y - 210), 8.0)

    def test_returns_not_found_for_blank_frame(self) -> None:
        config = ReticleTrackerConfig(search_radius_px=260)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)

        detection, _, _ = detect_center_reticle(image, config)

        self.assertFalse(detection.found)

    def test_rejects_arc_fragment_and_hud_clutter_false_positive(self) -> None:
        config = ReticleTrackerConfig(search_radius_px=260)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        add_arc_fragment_with_hud_clutter(
            image,
            center_x=603,
            center_y=377,
            radius=config.expected_outer_radius_px,
        )

        detection, _, _ = detect_center_reticle(image, config)

        self.assertFalse(detection.found)


if __name__ == "__main__":
    unittest.main()
