from __future__ import annotations

import unittest

import numpy as np

from app.actions.navigation_ocr import (
    _crop_warped_roi,
    _effective_cursor_index,
    _estimate_row_slot,
    _extract_fixed_row_lines,
    _find_best_target_match,
    _is_malformed_ocr_text,
    _locate_row_span_index,
    _ocr_row_with_fallbacks,
    _merge_slot_line_candidates,
    _normalize_nav_text,
    _prepare_direct_roi_for_ocr,
    _prepare_row_image_for_ocr,
    _prefer_ocr_line,
    _resolve_list_quad_points,
    _resolve_target_row_index,
    _resolve_output_region,
    _save_iteration_debug_artifacts,
    _stack_row_images_for_ocr,
    _text_similarity,
    _warp_list_quad,
    OcrNavConfig,
    OcrLine,
)
from unittest.mock import patch


class TestNavigationOcrHelpers(unittest.TestCase):
    def test_normalize_nav_text_collapses_spacing_and_symbols(self) -> None:
        self.assertEqual(_normalize_nav_text("  Abraham-Lincoln  "), "abrahamlincoln")
        self.assertEqual(_normalize_nav_text("HIP  20277  A 1"), "hip 20277 a 1")

    def test_text_similarity_prefers_exact_and_close_matches(self) -> None:
        self.assertEqual(_text_similarity("abraham lincoln", "abraham lincoln"), 1.0)
        self.assertGreater(_text_similarity("abraham lincoln", "abraham lincln"), 0.8)
        self.assertLess(_text_similarity("abraham lincoln", "earth"), 0.3)

    def test_find_best_target_match_uses_best_matching_visible_line(self) -> None:
        lines = [
            OcrLine(text="Alpha Centauri", confidence=92.0, bbox=(0, 0, 10, 10)),
            OcrLine(text="Abraham Lincln", confidence=85.0, bbox=(0, 15, 10, 10)),
            OcrLine(text="LTT 1349", confidence=88.0, bbox=(0, 30, 10, 10)),
        ]

        match = _find_best_target_match(lines, "abraham lincoln", 0.72)

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match[0], 1)
        line = match[1]
        self.assertEqual(line.text, "Abraham Lincln")

    def test_resolve_output_region_maps_absolute_window_to_monitor_relative_region(self) -> None:
        with patch(
            "app.actions.navigation_ocr._enumerate_display_monitors",
            return_value=[(0, 0, 1920, 1080, True), (1920, 0, 3840, 1080, False)],
        ):
            output_index, region = _resolve_output_region((1920, 0, 1920, 1032))

        self.assertEqual(output_index, 1)
        self.assertEqual(region, (0, 0, 1920, 1032))

    def test_effective_cursor_index_is_top_for_first_page_and_bottom_for_scrolling_pages(self) -> None:
        self.assertEqual(_effective_cursor_index(0, 11), 0)
        self.assertEqual(_effective_cursor_index(1, 11), 10)

    def test_estimate_row_slot_maps_bbox_center_to_expected_row_grid(self) -> None:
        line = OcrLine(text="HIP 17189", confidence=90.0, bbox=(100, 92, 200, 20))
        self.assertEqual(_estimate_row_slot(line, roi_height=330, expected_rows=11), 3)

    def test_resolve_target_row_index_uses_order_on_first_page(self) -> None:
        line = OcrLine(text="TOGETHER AN OCEAN Y8M-8XZ", confidence=90.0, bbox=(100, 200, 200, 20))
        self.assertEqual(
            _resolve_target_row_index(
                page_index=0,
                target_order_index=5,
                target_line=line,
                roi_height=392,
                expected_rows=11,
            ),
            5,
        )

    def test_resolve_target_row_index_does_not_round_into_next_row_on_scrolled_page(self) -> None:
        line = OcrLine(
            text="ORBITAL CONSTRUCTION SITE: PARISE GATEWAY",
            confidence=88.0,
            bbox=(0, 230, 200, 33),
        )
        self.assertEqual(
            _resolve_target_row_index(
                page_index=1,
                target_order_index=7,
                target_line=line,
                roi_height=361,
                expected_rows=11,
            ),
            7,
        )

    def test_is_malformed_ocr_text_rejects_tsv_blob_like_text(self) -> None:
        self.assertTrue(_is_malformed_ocr_text("MEL\n5\t1\t1\t1\t5\t2\t471\t315"))
        self.assertFalse(_is_malformed_ocr_text("HIP 17189"))

    def test_warp_list_quad_flattens_trapezoid_to_expected_size(self) -> None:
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        flattened = _warp_list_quad(
            frame,
            (
                (20, 10),
                (24, 90),
                (120, 14),
                (126, 86),
            ),
        )

        self.assertGreater(flattened.shape[0], 70)
        self.assertGreater(flattened.shape[1], 95)

    def test_prepare_direct_roi_for_ocr_returns_grayscale_image(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        processed = _prepare_direct_roi_for_ocr(image)

        self.assertEqual(len(processed.shape), 2)
        self.assertGreater(processed.shape[0], 20)
        self.assertGreater(processed.shape[1], 30)

    def test_extract_fixed_row_lines_returns_expected_row_count(self) -> None:
        text_roi = np.zeros((110, 100, 3), dtype=np.uint8)
        processed_roi = np.zeros((220, 200), dtype=np.uint8)

        batched_lines = [
            OcrLine(text="A", confidence=95.0, bbox=(0, 0, 100, 10)),
            OcrLine(text="", confidence=0.0, bbox=(0, 10, 100, 10)),
            OcrLine(text="C", confidence=90.0, bbox=(0, 20, 100, 10)),
        ] + [OcrLine(text="", confidence=0.0, bbox=(0, index * 10, 100, 10)) for index in range(3, 11)]

        with patch("app.actions.navigation_ocr._ocr_row_batch_with_fallbacks", return_value=batched_lines):
            lines = _extract_fixed_row_lines(
                text_roi=text_roi,
                processed_roi=processed_roi,
                tesseract_path="tesseract",
                expected_rows=11,
                highlighted_row_index=10,
                use_direct_roi_ocr=True,
            )

        self.assertEqual(len(lines), 11)
        self.assertEqual(lines[0].text, "A")
        self.assertEqual(lines[2].text, "C")
        self.assertEqual(lines[10].bbox[1], 100)

    def test_ocr_row_with_fallbacks_uses_secondary_pipeline_when_primary_is_empty(self) -> None:
        color_row = np.zeros((10, 40, 3), dtype=np.uint8)
        processed_row = np.zeros((20, 80), dtype=np.uint8)

        with patch("app.actions.navigation_ocr._ocr_single_row", side_effect=["", "TARGET"]):
            text = _ocr_row_with_fallbacks(
                color_row=color_row,
                processed_row=processed_row,
                highlighted=False,
                use_direct_roi_ocr=True,
                tesseract_path="tesseract",
                preferred_image=np.zeros((24, 96), dtype=np.uint8),
            )

        self.assertEqual(text, "TARGET")

    def test_prepare_row_image_for_ocr_handles_highlighted_rows_differently(self) -> None:
        color_row = np.zeros((10, 40, 3), dtype=np.uint8)
        color_row[:] = (255, 180, 50)
        processed_row = np.zeros((20, 80), dtype=np.uint8)

        normal = _prepare_row_image_for_ocr(color_row, processed_row, highlighted=False, use_direct_roi_ocr=True)
        highlighted = _prepare_row_image_for_ocr(color_row, processed_row, highlighted=True, use_direct_roi_ocr=True)

        self.assertEqual(normal.shape, highlighted.shape)
        self.assertFalse(np.array_equal(normal, highlighted))
        self.assertLess(normal.shape[1], int(40 * 2.4))

    def test_merge_slot_line_candidates_places_lines_into_expected_slots(self) -> None:
        lines = [
            OcrLine(text="ALPHA", confidence=80.0, bbox=(0, 0, 50, 6)),
            OcrLine(text="BETA", confidence=85.0, bbox=(0, 50, 50, 6)),
            OcrLine(text="GAMMA", confidence=90.0, bbox=(0, 100, 50, 6)),
        ]

        slots = _merge_slot_line_candidates(lines, roi_height=110, expected_rows=11, roi_width=100)

        self.assertEqual(len(slots), 11)
        self.assertEqual(slots[0].text, "ALPHA")
        self.assertEqual(slots[5].text, "BETA")
        self.assertEqual(slots[10].text, "GAMMA")

    def test_prefer_ocr_line_prefers_better_text_over_empty_or_weaker_text(self) -> None:
        empty = OcrLine(text="", confidence=0.0, bbox=(0, 0, 10, 10))
        weak = OcrLine(text="A", confidence=90.0, bbox=(0, 0, 10, 10))
        strong = OcrLine(text="HILDEBRANDT REFINERY", confidence=60.0, bbox=(0, 0, 10, 10))

        self.assertTrue(_prefer_ocr_line(strong, empty))
        self.assertTrue(_prefer_ocr_line(strong, weak))

    def test_ocr_nav_config_defaults_to_calibrated_projection_quad(self) -> None:
        self.assertEqual(
            OcrNavConfig().list_quad_points,
            (
                (545, 457),
                (569, 838),
                (1280, 426),
                (1289, 767),
            ),
        )
        self.assertEqual(
            OcrNavConfig().scrolled_list_quad_points,
            (
                (547, 467),
                (569, 848),
                (1280, 435),
                (1290, 775),
            ),
        )

    def test_resolve_list_quad_points_uses_scrolled_quad_after_first_page(self) -> None:
        first_quad = ((1, 1), (1, 2), (3, 1), (3, 2))
        next_quad = ((4, 4), (4, 5), (6, 4), (6, 5))

        self.assertEqual(_resolve_list_quad_points(0, first_quad, next_quad), first_quad)
        self.assertEqual(_resolve_list_quad_points(2, first_quad, next_quad), next_quad)

    def test_crop_warped_roi_trims_right_twenty_percent(self) -> None:
        image = np.zeros((10, 100, 3), dtype=np.uint8)

        cropped = _crop_warped_roi(image, 0.0, 0.80)

        self.assertEqual(cropped.shape, (10, 80, 3))

    def test_save_iteration_debug_artifacts_can_skip_debug_output(self) -> None:
        context = type("C", (), {"vision": None})()

        debug = _save_iteration_debug_artifacts(
            context=context,
            page_index=0,
            frame=np.zeros((10, 10, 3), dtype=np.uint8),
            roi=np.zeros((12, 8, 3), dtype=np.uint8),
            text_roi=np.zeros((12, 8, 3), dtype=np.uint8),
            text_mask=np.zeros((12, 8), dtype=np.uint8),
            processed=np.zeros((12, 8), dtype=np.uint8),
            lines=[],
            save_debug_artifacts=False,
        )

        self.assertEqual(debug, {"roi_height": 12})

    def test_stack_row_images_for_ocr_tracks_spans(self) -> None:
        rows = [np.zeros((10, 20), dtype=np.uint8), np.zeros((12, 30), dtype=np.uint8)]

        stacked, spans = _stack_row_images_for_ocr(rows)

        self.assertEqual(len(spans), 2)
        self.assertEqual(stacked.shape[1], 50)
        self.assertLess(spans[0][1], spans[1][0])

    def test_locate_row_span_index_maps_detected_line_to_correct_row(self) -> None:
        line = OcrLine(text="TARGET", confidence=90.0, bbox=(0, 35, 20, 10))
        spans = [(0, 20), (30, 50), (60, 80)]

        self.assertEqual(_locate_row_span_index(line, spans), 1)


if __name__ == "__main__":
    unittest.main()
