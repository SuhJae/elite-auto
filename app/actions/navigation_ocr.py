from __future__ import annotations

import csv
import ctypes
import io
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from ctypes import wintypes

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.actions.starport_buy import build_standalone_context
from app.adapters.capture_dxcam import DxcamCapture
from app.config import AppConfig
from app.domain.context import Context
from app.domain.result import Result


# Edit these values for standalone testing of this file.
STANDALONE_CONFIG_PATH: str | None = None
# STANDALONE_TARGET_NAME = "TOGETHER AN OCEAN Y8M-8XZ"
# STANDALONE_TARGET_NAME = "ORBITAL CONSTRUCTION SITE: PARISE GATEWAY"
STANDALONE_TARGET_NAME = "HILDEBRANDT REFINERY"
# STANDALONE_TARGET_NAME = "SYNUEFAI QT-O D7-70"
STANDALONE_WINDOW_TITLE = "Elite Dangerous"
STANDALONE_START_DELAY_SECONDS = 3.0
STANDALONE_LIST_REGION = (0.2208, 0.4186, 0.4042, 0.3634)
STANDALONE_LIST_QUAD = (
    (545, 457),
    (569, 838),
    (1280, 426),
    (1289, 767),
)
STANDALONE_SCROLLED_LIST_QUAD = (
    (547, 467),
    (569, 848),
    (1280, 435),
    (1290, 775),
)
STANDALONE_MAX_SCAN_PAGES = 12
STANDALONE_INITIAL_SCROLL_TRANSITION_PRESSES = 20
STANDALONE_SCROLL_PAGE_STEP_PRESSES = 10
STANDALONE_CAPTURE_SETTLE_SECONDS = 0.3
STANDALONE_MOVE_INTERVAL_SECONDS = 0.2
STANDALONE_MATCH_THRESHOLD = 0.72
STANDALONE_TESSERACT_PATH: str | None = None
STANDALONE_EXPECTED_VISIBLE_ROWS = 11
STANDALONE_USE_DIRECT_ROI_OCR = True
STANDALONE_WARP_CROP_LEFT_FRACTION = 0.0
STANDALONE_WARP_CROP_RIGHT_FRACTION = 0.80
STANDALONE_SAVE_DEBUG_ARTIFACTS = True

LEFT_PANEL_GUI_FOCUS = 2


@dataclass(slots=True)
class OcrLine:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class PreparedRowOcr:
    bbox: tuple[int, int, int, int]
    preferred_image: np.ndarray
    alternate_image: np.ndarray


@dataclass(slots=True)
class OcrNavTimings:
    capture_settle_seconds: float = 0.3
    move_interval_seconds: float = 0.2


@dataclass(slots=True)
class OcrNavConfig:
    window_title_substring: str = "Elite Dangerous"
    list_region_fractions: tuple[float, float, float, float] = (0.0, 0.16, 0.34, 0.76)
    list_quad_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None = (
        (545, 457),
        (569, 838),
        (1280, 426),
        (1289, 767),
    )
    scrolled_list_quad_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None = (
        (547, 467),
        (569, 848),
        (1280, 435),
        (1290, 775),
    )
    max_scan_pages: int = 12
    initial_scroll_transition_presses: int = 20
    scroll_page_step_presses: int = 10
    expected_visible_rows: int = 11
    match_threshold: float = 0.72
    tesseract_path: str | None = None
    use_direct_roi_ocr: bool = True
    warp_crop_left_fraction: float = 0.0
    warp_crop_right_fraction: float = 0.80
    save_debug_artifacts: bool = False


@dataclass(slots=True)
class MoveCursorToNavTarget:
    """Use OCR/debug capture to move the nav cursor onto a named target."""

    target_name: str
    timings: OcrNavTimings = field(default_factory=OcrNavTimings)
    config: OcrNavConfig = field(default_factory=OcrNavConfig)

    name = "move_cursor_to_nav_target"

    def run(self, context: Context) -> Result:
        ship_control = context.ship_control
        capture = context.capture
        vision = context.vision
        if ship_control is None or capture is None:
            return Result.fail("Navigation OCR requires both ship control and capture.")

        state = context.state_reader.snapshot()
        if state.gui_focus != LEFT_PANEL_GUI_FOCUS:
            return Result.fail(
                "Navigation OCR expects the left panel to already be open.",
                debug={"gui_focus": state.gui_focus, "expected_gui_focus": LEFT_PANEL_GUI_FOCUS},
            )

        target_key = _normalize_nav_text(self.target_name)
        if not target_key:
            return Result.fail("A target name is required.", debug={"target_name": self.target_name})

        tesseract_path = _find_tesseract(self.config.tesseract_path)
        if tesseract_path is None:
            return Result.fail(
                "Tesseract OCR was not found. Install Tesseract or set a path for navigation OCR.",
                debug={
                    "target_name": self.target_name,
                    "suggested_path": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                },
            )

        context.logger.info(
            "Starting navigation OCR scan",
            extra={
                "target_name": self.target_name,
                "tesseract_path": tesseract_path,
                "window_title_substring": self.config.window_title_substring,
                "list_region_fractions": self.config.list_region_fractions,
                "list_quad_points": self.config.list_quad_points,
                "scrolled_list_quad_points": self.config.scrolled_list_quad_points,
                "max_scan_pages": self.config.max_scan_pages,
                "initial_scroll_transition_presses": self.config.initial_scroll_transition_presses,
                "scroll_page_step_presses": self.config.scroll_page_step_presses,
                "use_direct_roi_ocr": self.config.use_direct_roi_ocr,
                "warp_crop_left_fraction": self.config.warp_crop_left_fraction,
                "warp_crop_right_fraction": self.config.warp_crop_right_fraction,
            },
        )

        window_capture_region = _find_window_client_region(self.config.window_title_substring)
        if window_capture_region is None:
            return Result.fail(
                "Could not locate the Elite Dangerous window for OCR capture.",
                debug={"window_title_substring": self.config.window_title_substring},
            )

        output_index, output_relative_region = _resolve_output_region(window_capture_region)
        if output_index is None or output_relative_region is None:
            return Result.fail(
                "Could not map the Elite Dangerous window onto a dxcam output.",
                debug={"window_capture_region": window_capture_region},
            )
        if isinstance(capture, DxcamCapture):
            capture.set_output_index(output_index)

        context.logger.info(
            "Navigation OCR window resolved",
            extra={
                "window_title_substring": self.config.window_title_substring,
                "window_capture_region": window_capture_region,
                "output_index": output_index,
                "output_relative_region": output_relative_region,
            },
        )

        total_down_presses = 0
        for page_index in range(self.config.max_scan_pages + 1):
            phase = "top_page" if page_index == 0 else "scrolling"
            if page_index == 1:
                _press_down_many(
                    ship_control,
                    self.config.initial_scroll_transition_presses,
                    self.timings.move_interval_seconds,
                )
                total_down_presses += self.config.initial_scroll_transition_presses
            elif page_index > 1:
                _press_down_many(
                    ship_control,
                    self.config.scroll_page_step_presses,
                    self.timings.move_interval_seconds,
                )
                total_down_presses += self.config.scroll_page_step_presses

            lines, debug_artifacts = _scan_nav_page(
                context=context,
                capture=capture,
                tesseract_path=tesseract_path,
                capture_region=output_relative_region,
                list_region_fractions=self.config.list_region_fractions,
                list_quad_points=self.config.list_quad_points,
                scrolled_list_quad_points=self.config.scrolled_list_quad_points,
                use_direct_roi_ocr=self.config.use_direct_roi_ocr,
                expected_rows=self.config.expected_visible_rows,
                highlighted_row_index=_effective_cursor_index(page_index, self.config.expected_visible_rows),
                page_index=page_index,
                settle_seconds=self.timings.capture_settle_seconds,
                warp_crop_left_fraction=self.config.warp_crop_left_fraction,
                warp_crop_right_fraction=self.config.warp_crop_right_fraction,
                save_debug_artifacts=self.config.save_debug_artifacts,
            )
            visible_texts = [line.text for line in lines]
            target_match = _find_best_target_match(lines, target_key, self.config.match_threshold) if lines else None
            target_order_index = target_match[0] if target_match is not None else None
            target_line = target_match[1] if target_match is not None else None
            target_index = _resolve_target_row_index(
                page_index=page_index,
                target_order_index=target_order_index,
                target_line=target_line,
                roi_height=debug_artifacts["roi_height"],
                expected_rows=self.config.expected_visible_rows,
            )
            best_match = target_line.text if target_line is not None else None
            effective_cursor_index = _effective_cursor_index(page_index, self.config.expected_visible_rows)

            overlap_match = None

            context.logger.info(
                "Navigation OCR iteration",
                extra={
                    "page_index": page_index,
                    "phase": phase,
                    "total_down_presses": total_down_presses,
                    "target_name": self.target_name,
                    "visible_line_count": len(visible_texts),
                    "visible_preview": visible_texts[:5],
                    "effective_cursor_index": effective_cursor_index,
                    "target_order_index": target_order_index,
                    "target_index": target_index,
                    "expected_visible_rows": self.config.expected_visible_rows,
                    "best_match": best_match,
                    "overlap_match": overlap_match,
                    "debug_artifacts": debug_artifacts,
                },
            )

            if not lines:
                continue

            if target_index is None:
                continue

            move_direction = "down" if target_index > effective_cursor_index else "up"
            move_count = abs(target_index - effective_cursor_index)
            context.logger.info(
                "Navigation OCR target located",
                extra={
                    "page_index": page_index,
                    "phase": phase,
                    "target_order_index": target_order_index,
                    "target_index": target_index,
                    "effective_cursor_index": effective_cursor_index,
                    "move_direction": move_direction,
                    "move_count": move_count,
                    "best_match": best_match,
                },
            )

            _press_many(ship_control, move_direction, move_count, self.timings.move_interval_seconds)

            return Result.ok(
                "Navigation cursor moved onto target.",
                debug={
                    "target_name": self.target_name,
                    "page_index": page_index,
                    "phase": phase,
                    "visible_preview": visible_texts[:5],
                    "visible_line_count": len(visible_texts),
                    "target_order_index": target_order_index,
                    "target_index": target_index,
                    "effective_cursor_index": effective_cursor_index,
                    "best_match": best_match,
                    "overlap_match": overlap_match,
                    "move_direction": move_direction,
                    "move_count": move_count,
                    "debug_artifacts": debug_artifacts,
                },
            )

        return Result.fail(
            "Navigation OCR scan did not find the target within the scan limit.",
            debug={"target_name": self.target_name, "max_scan_pages": self.config.max_scan_pages},
        )


def _find_tesseract(configured_path: str | None) -> str | None:
    candidates = [
        configured_path,
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def _press_down_many(ship_control: Any, count: int, move_interval_seconds: float) -> None:
    _press_many(ship_control, "down", count, move_interval_seconds)


def _press_many(ship_control: Any, direction: str, count: int, move_interval_seconds: float) -> None:
    for _ in range(max(0, count)):
        ship_control.ui_select(direction)
        time.sleep(move_interval_seconds)


def _scan_nav_page(
    context: Context,
    capture: Any,
    tesseract_path: str,
    capture_region: tuple[int, int, int, int],
    list_region_fractions: tuple[float, float, float, float],
    list_quad_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None,
    scrolled_list_quad_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None,
    use_direct_roi_ocr: bool,
    expected_rows: int,
    highlighted_row_index: int,
    page_index: int,
    settle_seconds: float,
    warp_crop_left_fraction: float,
    warp_crop_right_fraction: float,
    save_debug_artifacts: bool,
) -> tuple[list[OcrLine], dict[str, Any]]:
    time.sleep(settle_seconds)
    frame = capture.grab(region=capture_region)
    quad_points = _resolve_list_quad_points(page_index, list_quad_points, scrolled_list_quad_points)
    if quad_points is not None:
        roi = _warp_list_quad(frame, quad_points)
        roi = _crop_warped_roi(roi, warp_crop_left_fraction, warp_crop_right_fraction)
    else:
        list_region = _scale_region(frame, list_region_fractions)
        roi = _crop_region(frame, list_region)
    text_roi = roi.copy()
    text_mask = _build_ui_text_mask(roi)
    processed = _prepare_direct_roi_for_ocr(text_roi) if use_direct_roi_ocr else _preprocess_for_ocr(text_roi, text_mask)
    lines = _extract_fixed_row_lines(
        text_roi=text_roi,
        processed_roi=processed,
        tesseract_path=tesseract_path,
        expected_rows=expected_rows,
        highlighted_row_index=highlighted_row_index,
        use_direct_roi_ocr=use_direct_roi_ocr,
    )
    debug_artifacts = _save_iteration_debug_artifacts(
        context=context,
        page_index=page_index,
        frame=frame,
        roi=roi,
        text_roi=text_roi,
        text_mask=text_mask,
        processed=processed,
        lines=lines,
        save_debug_artifacts=save_debug_artifacts,
    )
    return lines, debug_artifacts


def _warp_list_quad(
    frame: np.ndarray,
    quad_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    top_left, bottom_left, top_right, bottom_right = quad_points
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    top_width = float(np.linalg.norm(np.array(top_right) - np.array(top_left)))
    bottom_width = float(np.linalg.norm(np.array(bottom_right) - np.array(bottom_left)))
    left_height = float(np.linalg.norm(np.array(bottom_left) - np.array(top_left)))
    right_height = float(np.linalg.norm(np.array(bottom_right) - np.array(top_right)))

    width = max(1, int(round((top_width + bottom_width) / 2.0)))
    height = max(1, int(round((left_height + right_height) / 2.0)))
    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, transform, (width, height))


def _resolve_list_quad_points(
    page_index: int,
    first_page_quad_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None,
    scrolled_quad_points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if page_index <= 0:
        return first_page_quad_points
    return scrolled_quad_points or first_page_quad_points


def _crop_warped_roi(image: np.ndarray, left_fraction: float, right_fraction: float) -> np.ndarray:
    height, width = image.shape[:2]
    clamped_left = max(0.0, min(0.95, left_fraction))
    clamped_right = max(clamped_left + 0.01, min(1.0, right_fraction))
    x0 = int(round(width * clamped_left))
    x1 = int(round(width * clamped_right))
    if x1 <= x0 or x0 >= width:
        return image.copy()
    return image[:, x0:min(width, x1)].copy()


def _scroll_overlap_matches(previous_lines: list[OcrLine] | None, current_lines: list[OcrLine]) -> bool | None:
    if not previous_lines or not current_lines:
        return None

    previous_bottom = _normalize_nav_text(previous_lines[-1].text)
    current_top = _normalize_nav_text(current_lines[0].text)
    if not previous_bottom or not current_top:
        return None
    return previous_bottom == current_top


def _effective_cursor_index(page_index: int, expected_visible_rows: int) -> int:
    if page_index <= 0:
        return 0
    return max(0, expected_visible_rows - 1)


def _find_window_client_region(window_title_substring: str) -> tuple[int, int, int, int] | None:
    user32 = ctypes.windll.user32
    found_hwnd: list[int] = []
    title_match = _normalize_window_title(window_title_substring)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_windows_proc(hwnd: int, lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, len(buffer))
        title = buffer.value.strip()
        if title_match not in _normalize_window_title(title):
            return True

        found_hwnd.append(hwnd)
        return False

    user32.EnumWindows(enum_windows_proc, 0)
    if not found_hwnd:
        return None

    hwnd = found_hwnd[0]
    if user32.IsIconic(hwnd):
        return None

    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None

    top_left = wintypes.POINT(rect.left, rect.top)
    bottom_right = wintypes.POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        return None

    width = bottom_right.x - top_left.x
    height = bottom_right.y - top_left.y
    if width <= 0 or height <= 0:
        return None

    return top_left.x, top_left.y, width, height


def _normalize_window_title(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _resolve_output_region(
    absolute_region: tuple[int, int, int, int],
) -> tuple[int | None, tuple[int, int, int, int] | None]:
    x, y, width, height = absolute_region
    center_x = x + (width // 2)
    center_y = y + (height // 2)

    monitors = _enumerate_display_monitors()
    for output_index, monitor in enumerate(monitors):
        left, top, right, bottom, _is_primary = monitor
        if left <= center_x < right and top <= center_y < bottom:
            return output_index, (x - left, y - top, width, height)
    return None, None


def _enumerate_display_monitors() -> list[tuple[int, int, int, int, bool]]:
    user32 = ctypes.windll.user32
    monitors: list[tuple[int, int, int, int, bool]] = []

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    monitor_enum_proc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    def callback(hmonitor: int, hdc: int, rect_ptr: Any, lparam: int) -> bool:
        rect = rect_ptr.contents
        monitor_info = MONITORINFOEXW()
        monitor_info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if not user32.GetMonitorInfoW(hmonitor, ctypes.byref(monitor_info)):
            return True
        is_primary = bool(monitor_info.dwFlags & 1)
        monitors.append((rect.left, rect.top, rect.right, rect.bottom, is_primary))
        return True

    user32.EnumDisplayMonitors(0, 0, monitor_enum_proc(callback), 0)
    monitors.sort(key=lambda monitor: (not monitor[4], monitor[1], monitor[0]))
    return monitors


def _scale_region(frame: np.ndarray, fractions: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x_fraction, y_fraction, width_fraction, height_fraction = fractions
    x = int(width * x_fraction)
    y = int(height * y_fraction)
    region_width = int(width * width_fraction)
    region_height = int(height * height_fraction)
    return x, y, region_width, region_height


def _crop_region(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = region
    return frame[y : y + height, x : x + width].copy()


def _build_ui_text_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cyan_mask = cv2.inRange(hsv, np.array([80, 60, 110]), np.array([125, 255, 255]))
    orange_mask = cv2.inRange(hsv, np.array([8, 60, 110]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(cyan_mask, orange_mask)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_wide = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_wide, iterations=1)
    return mask


def _derive_text_roi(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = _build_ui_text_mask(roi)
    height, width = roi.shape[:2]

    row_density = np.count_nonzero(mask > 0, axis=1)
    col_density = np.count_nonzero(mask > 0, axis=0)

    row_start_search = int(height * 0.18)
    col_start_search = int(width * 0.45)
    row_threshold = max(8, int(width * 0.01))
    col_threshold = max(6, int(height * 0.02))

    active_rows = np.where(row_density[row_start_search:] > row_threshold)[0]
    active_cols = np.where(col_density[col_start_search:] > col_threshold)[0]

    if active_rows.size == 0 or active_cols.size == 0:
        fallback_x = int(width * 0.58)
        fallback_y = int(height * 0.24)
        fallback_w = max(1, int(width * 0.37))
        fallback_h = max(1, int(height * 0.66))
        return _crop_region(roi, (fallback_x, fallback_y, fallback_w, fallback_h)), _crop_region(
            mask, (fallback_x, fallback_y, fallback_w, fallback_h)
        )

    y0 = row_start_search + int(active_rows[0])
    y1 = row_start_search + int(active_rows[-1]) + 1
    x0 = col_start_search + int(active_cols[0])
    x1 = col_start_search + int(active_cols[-1]) + 1

    pad_x = 10
    pad_y = 6
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    region = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    return _crop_region(roi, region), _crop_region(mask, region)


def _preprocess_for_ocr(image: np.ndarray, text_mask: np.ndarray | None = None) -> np.ndarray:
    enlarged = cv2.resize(image, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    if text_mask is None:
        mask = _build_ui_text_mask(enlarged)
    else:
        mask = cv2.resize(text_mask, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Strengthen thin glyphs while keeping rows separate.
    thresholded = cv2.dilate(thresholded, kernel_small, iterations=1)
    return thresholded


def _prepare_direct_roi_for_ocr(image: np.ndarray) -> np.ndarray:
    enlarged = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _extract_slot_lines_from_full_roi(
    color_roi: np.ndarray,
    processed_roi: np.ndarray,
    tesseract_path: str,
    expected_rows: int,
) -> list[OcrLine]:
    direct_lines = _extract_ocr_lines_from_image(color_roi, tesseract_path, psm="6", min_confidence=25.0)
    processed_lines = _extract_ocr_lines_from_image(processed_roi, tesseract_path, psm="6", min_confidence=25.0)
    merged_lines = _merge_slot_line_candidates(
        direct_lines + processed_lines,
        roi_height=processed_roi.shape[0],
        expected_rows=expected_rows,
        roi_width=processed_roi.shape[1],
    )
    return merged_lines


def _extract_fixed_row_lines(
    text_roi: np.ndarray,
    processed_roi: np.ndarray,
    tesseract_path: str,
    expected_rows: int,
    highlighted_row_index: int,
    use_direct_roi_ocr: bool,
) -> list[OcrLine]:
    color_height, color_width = text_roi.shape[:2]
    processed_height, processed_width = processed_roi.shape[:2]
    color_row_height = max(1.0, color_height / max(expected_rows, 1))
    processed_row_height = max(1.0, processed_height / max(expected_rows, 1))
    prepared_rows: list[PreparedRowOcr] = []

    for row_index in range(expected_rows):
        color_y0 = int(round(row_index * color_row_height))
        color_y1 = int(round((row_index + 1) * color_row_height)) if row_index < expected_rows - 1 else color_height
        if color_y1 <= color_y0:
            color_y1 = min(color_height, color_y0 + 1)

        processed_y0 = int(round(row_index * processed_row_height))
        processed_y1 = (
            int(round((row_index + 1) * processed_row_height)) if row_index < expected_rows - 1 else processed_height
        )
        if processed_y1 <= processed_y0:
            processed_y1 = min(processed_height, processed_y0 + 1)

        color_row = text_roi[color_y0:color_y1, :].copy()
        processed_row = processed_roi[processed_y0:processed_y1, :].copy()
        preferred_image = _prepare_row_image_for_ocr(
            color_row=color_row,
            processed_row=processed_row,
            highlighted=(row_index == highlighted_row_index),
            use_direct_roi_ocr=use_direct_roi_ocr,
        )
        alternate_image = _prepare_row_image_for_ocr(
            color_row=color_row,
            processed_row=processed_row,
            highlighted=(row_index == highlighted_row_index),
            use_direct_roi_ocr=not use_direct_roi_ocr,
        )
        prepared_rows.append(
            PreparedRowOcr(
                bbox=(0, color_y0, color_width, max(1, color_y1 - color_y0)),
                preferred_image=preferred_image,
                alternate_image=alternate_image,
            )
        )
    return _ocr_row_batch_with_fallbacks(prepared_rows, tesseract_path)


def _ocr_row_batch_with_fallbacks(prepared_rows: list[PreparedRowOcr], tesseract_path: str) -> list[OcrLine]:
    if not prepared_rows:
        return []

    preferred_lines = _ocr_prepared_rows_batch(
        [prepared_row.preferred_image for prepared_row in prepared_rows],
        [prepared_row.bbox for prepared_row in prepared_rows],
        tesseract_path,
    )
    lines = list(preferred_lines)

    for index, (prepared_row, line) in enumerate(zip(prepared_rows, lines)):
        if not _should_retry_row_with_single_line_ocr(line, prepared_row.bbox[2]):
            continue
        retried_line = _ocr_prepared_row_with_fallbacks(prepared_row, tesseract_path)
        if _prefer_ocr_line(retried_line, line):
            lines[index] = retried_line

    recognized_count = sum(1 for line in lines if line.text)
    minimum_acceptable_rows = max(3, len(prepared_rows) // 4)
    if recognized_count >= minimum_acceptable_rows:
        return lines

    alternate_lines = _ocr_prepared_rows_batch(
        [prepared_row.alternate_image for prepared_row in prepared_rows],
        [prepared_row.bbox for prepared_row in prepared_rows],
        tesseract_path,
    )
    for index, alternate_line in enumerate(alternate_lines):
        if _prefer_ocr_line(alternate_line, lines[index]):
            lines[index] = alternate_line
    return lines


def _ocr_prepared_rows_batch(
    row_images: list[np.ndarray],
    row_bboxes: list[tuple[int, int, int, int]],
    tesseract_path: str,
) -> list[OcrLine]:
    if not row_images:
        return []

    stacked_image, row_spans = _stack_row_images_for_ocr(row_images)
    detected_lines = _extract_ocr_lines_from_image(stacked_image, tesseract_path, psm="11", min_confidence=20.0)
    results = [OcrLine(text="", confidence=0.0, bbox=bbox) for bbox in row_bboxes]

    for detected_line in detected_lines:
        row_index = _locate_row_span_index(detected_line, row_spans)
        if row_index is None:
            continue
        row_bbox = row_bboxes[row_index]
        candidate = OcrLine(
            text=detected_line.text,
            confidence=detected_line.confidence,
            bbox=(
                detected_line.bbox[0],
                row_bbox[1],
                detected_line.bbox[2],
                row_bbox[3],
            ),
        )
        if _prefer_ocr_line(candidate, results[row_index]):
            results[row_index] = candidate
    return results


def _ocr_prepared_row_with_fallbacks(prepared_row: PreparedRowOcr, tesseract_path: str) -> OcrLine:
    for image, psm in (
        (prepared_row.preferred_image, "7"),
        (prepared_row.preferred_image, "11"),
        (prepared_row.alternate_image, "7"),
    ):
        text = _ocr_single_row_once(image, tesseract_path, psm)
        if text:
            return OcrLine(
                text=text,
                confidence=100.0,
                bbox=prepared_row.bbox,
            )

    return OcrLine(
        text="",
        confidence=0.0,
        bbox=prepared_row.bbox,
    )


def _should_retry_row_with_single_line_ocr(line: OcrLine, row_width: int) -> bool:
    if not line.text:
        return True

    normalized = line.text.strip()
    words = [part for part in normalized.split() if part]
    normalized_key = _normalize_nav_text(normalized)
    allowed_single_words = {
        "unexplored",
    }
    common_suffix_words = {
        "folly",
        "legacy",
        "horizons",
        "refinery",
        "foundry",
        "gateway",
        "beacon",
    }

    if normalized_key in allowed_single_words:
        return False

    starts_far_right = line.bbox[0] >= int(row_width * 0.35)
    narrow_capture = line.bbox[2] <= int(row_width * 0.35)
    looks_clipped = normalized.startswith(("<", "&", "'", '"')) or normalized.endswith(("<", "&", "'", '"'))
    if len(words) <= 1 and (starts_far_right or looks_clipped or normalized_key in common_suffix_words):
        return True
    if starts_far_right and narrow_capture:
        return True
    if starts_far_right and len(words) <= 2:
        return True
    if looks_clipped and len(words) <= 2:
        return True
    return False


def _stack_row_images_for_ocr(row_images: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    max_width = max(image.shape[1] for image in row_images)
    border_x = 10
    border_y = 4
    row_gap = 12
    background_value = 255 if np.mean([float(image.mean()) for image in row_images]) >= 127.0 else 0
    total_height = sum((image.shape[0] + (border_y * 2)) for image in row_images) + (row_gap * max(0, len(row_images) - 1))
    stacked = np.full((total_height, max_width + (border_x * 2)), background_value, dtype=np.uint8)
    row_spans: list[tuple[int, int]] = []

    cursor_y = 0
    for row_image in row_images:
        row_height, row_width = row_image.shape[:2]
        band_top = cursor_y
        image_top = band_top + border_y
        image_left = border_x + ((max_width - row_width) // 2)
        stacked[image_top : image_top + row_height, image_left : image_left + row_width] = row_image
        band_bottom = band_top + (border_y * 2) + row_height
        row_spans.append((band_top, band_bottom))
        cursor_y = band_bottom + row_gap

    return stacked, row_spans


def _locate_row_span_index(line: OcrLine, row_spans: list[tuple[int, int]]) -> int | None:
    center_y = line.bbox[1] + (line.bbox[3] / 2.0)

    for index, (top, bottom) in enumerate(row_spans):
        if top <= center_y < bottom:
            return index

    best_index: int | None = None
    best_distance: float | None = None
    for index, (top, bottom) in enumerate(row_spans):
        span_center = (top + bottom) / 2.0
        distance = abs(center_y - span_center)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _ocr_row_with_fallbacks(
    color_row: np.ndarray,
    processed_row: np.ndarray,
    highlighted: bool,
    use_direct_roi_ocr: bool,
    tesseract_path: str,
    preferred_image: np.ndarray | None = None,
) -> str:
    candidates: list[str] = []

    if preferred_image is not None:
        text = _ocr_single_row(preferred_image, tesseract_path)
        if text:
            candidates.append(text)

    if use_direct_roi_ocr:
        processed_candidate = _prepare_row_image_for_ocr(
            color_row=color_row,
            processed_row=processed_row,
            highlighted=highlighted,
            use_direct_roi_ocr=False,
        )
        text = _ocr_single_row(processed_candidate, tesseract_path)
        if text:
            candidates.append(text)
    else:
        direct_candidate = _prepare_row_image_for_ocr(
            color_row=color_row,
            processed_row=processed_row,
            highlighted=highlighted,
            use_direct_roi_ocr=True,
        )
        text = _ocr_single_row(direct_candidate, tesseract_path)
        if text:
            candidates.append(text)

    if not candidates:
        return ""
    return max(candidates, key=_score_ocr_text_candidate)


def _prepare_row_image_for_ocr(
    color_row: np.ndarray,
    processed_row: np.ndarray,
    highlighted: bool,
    use_direct_roi_ocr: bool,
) -> np.ndarray:
    height, width = color_row.shape[:2]
    text_start = min(width - 1, max(0, int(width * 0.05)))
    text_end = min(width, max(text_start + 1, int(width * 0.96)))
    top_trim = max(0, int(height * 0.08))
    bottom_trim = min(height, max(top_trim + 1, int(height * 0.92)))
    row_color = color_row[top_trim:bottom_trim, text_start:text_end].copy()
    row_processed = processed_row[top_trim:bottom_trim, text_start:text_end].copy()

    if use_direct_roi_ocr:
        enlarged = cv2.resize(row_color, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if not highlighted:
            gray = cv2.bitwise_not(gray)
        return gray

    enlarged = cv2.resize(row_processed, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
    if highlighted:
        return cv2.bitwise_not(enlarged)
    return enlarged


def _ocr_single_row(image: np.ndarray, tesseract_path: str) -> str:
    candidates: list[str] = []
    for psm in ("7", "11"):
        text = _ocr_single_row_once(image, tesseract_path, psm)
        if text:
            candidates.append(text)

    if not candidates:
        return ""

    return max(candidates, key=_score_ocr_text_candidate)


def _ocr_single_row_once(image: np.ndarray, tesseract_path: str, psm: str) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_path = root / "ocr_row.png"
        if not cv2.imwrite(str(input_path), image):
            raise RuntimeError(f"Failed to write OCR row image: {input_path}")

        command = [
            tesseract_path,
            str(input_path),
            "stdout",
            "-l",
            "eng",
            "--oem",
            "1",
            "--psm",
            psm,
            "-c",
            "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -'&:/<>+",
            "quiet",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Tesseract OCR failed: {completed.stderr.strip() or completed.stdout.strip()}")

    text = completed.stdout.strip()
    if _is_malformed_ocr_text(text):
        return ""
    return text


def _score_ocr_text_candidate(text: str) -> tuple[int, int, int]:
    alnum_count = sum(character.isalnum() for character in text)
    letter_count = sum(character.isalpha() for character in text)
    punctuation_count = sum(not character.isalnum() and not character.isspace() for character in text)
    return (alnum_count, letter_count, -punctuation_count)


def _extract_ocr_lines(image: np.ndarray, tesseract_path: str) -> list[OcrLine]:
    return _extract_ocr_lines_from_image(image, tesseract_path, psm="6", min_confidence=20.0)


def _extract_ocr_lines_from_image(
    image: np.ndarray,
    tesseract_path: str,
    psm: str,
    min_confidence: float,
) -> list[OcrLine]:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_path = root / "ocr_input.png"
        output_base = root / "ocr_output"
        if not cv2.imwrite(str(input_path), image):
            raise RuntimeError(f"Failed to write OCR input image: {input_path}")

        command = [
            tesseract_path,
            str(input_path),
            str(output_base),
            "-l",
            "eng",
            "--oem",
            "1",
            "--psm",
            psm,
            "-c",
            "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -'&:/<>",
            "tsv",
            "quiet",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Tesseract OCR failed: {completed.stderr.strip() or completed.stdout.strip()}")

        tsv_path = output_base.with_suffix(".tsv")
        if not tsv_path.exists():
            raise RuntimeError(f"Tesseract did not produce TSV output: {tsv_path}")

        rows = list(csv.DictReader(io.StringIO(tsv_path.read_text(encoding="utf-8")), delimiter="\t"))

    grouped: dict[tuple[int, int, int, int], list[dict[str, str]]] = {}
    for row in rows:
        text = (row.get("text") or "").strip()
        conf_raw = (row.get("conf") or "").strip()
        if not text or conf_raw in {"", "-1"}:
            continue
        confidence = float(conf_raw)
        if confidence < min_confidence:
            continue
        if _is_malformed_ocr_text(text):
            continue
        line_key = (
            int(row.get("block_num") or 0),
            int(row.get("par_num") or 0),
            int(row.get("line_num") or 0),
            int(row.get("page_num") or 0),
        )
        grouped.setdefault(line_key, []).append(row)

    lines: list[OcrLine] = []
    for words in grouped.values():
        texts = [word["text"].strip() for word in words if word.get("text")]
        if not texts:
            continue
        combined_text = " ".join(texts)
        if _is_malformed_ocr_text(combined_text):
            continue
        lefts = [int(word["left"]) for word in words]
        tops = [int(word["top"]) for word in words]
        widths = [int(word["width"]) for word in words]
        heights = [int(word["height"]) for word in words]
        confidences = [max(0.0, float(word["conf"])) for word in words]
        x = min(lefts)
        y = min(tops)
        right = max(left + width for left, width in zip(lefts, widths))
        bottom = max(top + height for top, height in zip(tops, heights))
        lines.append(
            OcrLine(
                text=combined_text,
                confidence=sum(confidences) / max(len(confidences), 1),
                bbox=(x, y, right - x, bottom - y),
            )
        )

    lines.sort(key=lambda line: line.bbox[1])
    return lines


def _merge_slot_line_candidates(
    lines: list[OcrLine],
    roi_height: int,
    expected_rows: int,
    roi_width: int,
) -> list[OcrLine]:
    slots: list[OcrLine] = []
    for slot_index in range(expected_rows):
        slots.append(
            OcrLine(
                text="",
                confidence=0.0,
                bbox=(0, int(round(slot_index * (roi_height / max(expected_rows, 1)))), roi_width, max(1, int(round(roi_height / max(expected_rows, 1))))),
            )
        )

    for line in lines:
        slot_index = _estimate_row_slot(line, roi_height=roi_height, expected_rows=expected_rows)
        current = slots[slot_index]
        if _prefer_ocr_line(line, current):
            slots[slot_index] = line
    return slots


def _prefer_ocr_line(candidate: OcrLine, current: OcrLine) -> bool:
    if not current.text:
        return True
    candidate_score = _score_ocr_text_candidate(candidate.text)
    current_score = _score_ocr_text_candidate(current.text)
    if candidate_score != current_score:
        return candidate_score > current_score
    return candidate.confidence > current.confidence


def _find_best_target_match(lines: list[OcrLine], target_key: str, match_threshold: float) -> tuple[int, OcrLine] | None:
    best_match: tuple[int, OcrLine] | None = None
    best_score = 0.0
    for index, line in enumerate(lines):
        score = _text_similarity(target_key, _normalize_nav_text(line.text))
        if score >= match_threshold and score > best_score:
            best_score = score
            best_match = (index, line)
    return best_match


def _resolve_target_row_index(
    page_index: int,
    target_order_index: int | None,
    target_line: OcrLine | None,
    roi_height: int,
    expected_rows: int,
) -> int | None:
    if target_order_index is None or target_line is None:
        return None
    if page_index <= 0:
        return target_order_index
    return _estimate_row_slot(target_line, roi_height=roi_height, expected_rows=expected_rows)


def _estimate_row_slot(line: OcrLine, roi_height: int, expected_rows: int) -> int:
    row_height = max(1.0, roi_height / max(expected_rows, 1))
    center_y = line.bbox[1] + (line.bbox[3] / 2.0)
    slot = int(center_y / row_height)
    return max(0, min(expected_rows - 1, slot))


def _text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))
    return _levenshtein_ratio(left, right)


def _levenshtein_ratio(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    distance = previous[-1]
    return 1.0 - (distance / max(len(left), len(right)))


def _normalize_nav_text(value: str) -> str:
    normalized = "".join(character.lower() for character in value if character.isalnum() or character.isspace())
    return " ".join(normalized.split())


def _is_malformed_ocr_text(text: str) -> bool:
    if "\n" in text or "\t" in text:
        return True
    if len(text) > 120:
        return True
    digit_count = sum(character.isdigit() for character in text)
    if digit_count > max(8, len(text) // 3):
        return True
    return False


def _save_iteration_debug_artifacts(
    context: Context,
    page_index: int,
    frame: np.ndarray,
    roi: np.ndarray,
    text_roi: np.ndarray,
    text_mask: np.ndarray,
    processed: np.ndarray,
    lines: list[OcrLine],
    save_debug_artifacts: bool,
) -> dict[str, Any]:
    if not save_debug_artifacts or context.vision is None:
        return {"roi_height": int(roi.shape[0])}

    base_name = f"nav_ocr_page_{page_index:03d}"
    image_paths = {
        "full": str(context.vision.save_debug_snapshot(f"{base_name}_full", frame)),
        "roi": str(context.vision.save_debug_snapshot(f"{base_name}_roi", roi)),
        "text_roi": str(context.vision.save_debug_snapshot(f"{base_name}_text_roi", text_roi)),
        "text_mask": str(context.vision.save_debug_snapshot(f"{base_name}_text_mask", text_mask)),
        "processed": str(context.vision.save_debug_snapshot(f"{base_name}_processed", processed)),
    }
    text_path = context.debug_snapshot_dir / f"{base_name}_lines.txt"
    lines_payload = "\n".join(
        f"{index:02d} | conf={line.confidence:05.1f} | {line.text}"
        for index, line in enumerate(lines)
    )
    text_path.write_text(lines_payload or "(no OCR lines)", encoding="utf-8")
    image_paths["ocr_lines"] = str(text_path)
    image_paths["roi_height"] = int(text_roi.shape[0])
    return image_paths


def _main() -> int:
    config = AppConfig.from_json(STANDALONE_CONFIG_PATH) if STANDALONE_CONFIG_PATH else AppConfig.default()
    context = build_standalone_context(config)

    if STANDALONE_START_DELAY_SECONDS > 0:
        print(
            f"Warning: focusing game window. Starting move_cursor_to_nav_target in {STANDALONE_START_DELAY_SECONDS:.1f} seconds..."
        )
        time.sleep(STANDALONE_START_DELAY_SECONDS)

    action = MoveCursorToNavTarget(
        target_name=STANDALONE_TARGET_NAME,
        timings=OcrNavTimings(
            capture_settle_seconds=STANDALONE_CAPTURE_SETTLE_SECONDS,
            move_interval_seconds=STANDALONE_MOVE_INTERVAL_SECONDS,
        ),
        config=OcrNavConfig(
            window_title_substring=STANDALONE_WINDOW_TITLE,
            list_region_fractions=STANDALONE_LIST_REGION,
            list_quad_points=STANDALONE_LIST_QUAD,
            scrolled_list_quad_points=STANDALONE_SCROLLED_LIST_QUAD,
            max_scan_pages=STANDALONE_MAX_SCAN_PAGES,
            initial_scroll_transition_presses=STANDALONE_INITIAL_SCROLL_TRANSITION_PRESSES,
            scroll_page_step_presses=STANDALONE_SCROLL_PAGE_STEP_PRESSES,
            expected_visible_rows=STANDALONE_EXPECTED_VISIBLE_ROWS,
            match_threshold=STANDALONE_MATCH_THRESHOLD,
            tesseract_path=STANDALONE_TESSERACT_PATH,
            use_direct_roi_ocr=STANDALONE_USE_DIRECT_ROI_OCR,
            warp_crop_left_fraction=STANDALONE_WARP_CROP_LEFT_FRACTION,
            warp_crop_right_fraction=STANDALONE_WARP_CROP_RIGHT_FRACTION,
            save_debug_artifacts=STANDALONE_SAVE_DEBUG_ARTIFACTS,
        ),
    )
    result = action.run(context)
    print(result.reason)
    if result.debug:
        print("Debug:")
        print(result.format_debug())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
