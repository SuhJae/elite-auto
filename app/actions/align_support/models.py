from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from app.domain.protocols import Region


@dataclass(slots=True)
class AlignConfig:
    center_x: int = 731
    center_y: int = 818
    roi_size: int = 200
    compass_radius_px: float = 32.0
    compass_control_rotation_degrees: float = 0.0
    dynamic_center_enabled: bool = True
    center_dot_search_radius_px: float = 15.0
    center_dot_min_area: int = 2
    center_dot_max_area: int = 80
    center_dot_circle_center_max_distance_px: float = 8.0
    circle_detection_radius_tolerance_px: int = 8
    circle_detection_center_hint_tolerance_px: float = 12.0
    circle_detection_crop_padding_px: float = 14.0
    circle_detection_dp: float = 1.2
    circle_detection_param1: float = 80.0
    circle_detection_param2: float = 14.0
    circle_detection_min_axis_ratio: float = 0.68
    ring_blue_h_min: int = 85
    ring_blue_h_max: int = 130
    ring_blue_s_min: int = 40
    ring_blue_v_min: int = 70
    outer_ring_blue_h_min: int = 80
    outer_ring_blue_h_max: int = 135
    outer_ring_blue_s_min: int = 18
    outer_ring_blue_v_min: int = 35
    inner_ring_bgr_prototype: tuple[float, float, float] = (255.0, 80.0, 0.0)
    inner_ring_bgr_distance_threshold: float = 70.0
    outer_ring_bgr_prototype: tuple[float, float, float] = (210.0, 70.0, 10.0)
    outer_ring_bgr_distance_threshold: float = 75.0
    ring_min_area: int = 100
    inner_ring_outer_radius_offset_px: float = 8.0
    ring_score_inner_width_px: float = 2.0
    ring_score_outer_width_px: float = 2.5
    ring_score_angular_bins: int = 24
    ring_score_candidate_padding_px: int = 16
    inner_ring_template_radius_tolerance_px: int = 2
    inner_ring_template_padding_px: int = 8
    inner_ring_template_accept_score: float = 0.35
    inner_ring_tracking_only: bool = True
    inner_ring_center_search_radius_px: int = 16
    inner_ring_center_search_min_score: float = 0.18
    control_target_fps: int = 60
    center_tolerance_px: int = 4
    alignment_tolerance_px: float = 2.0
    axis_alignment_tolerance_px: float = 1.0
    confirmation_axis_tolerance_px: float = 5.0
    confirmation_distance_tolerance_px: float = 6.0
    alignment_dwell_seconds: float = 0.20
    near_center_consensus_enabled: bool = True
    near_center_consensus_trigger_distance_px: float = 6.0
    near_center_consensus_pause_seconds: float = 0.0
    near_center_consensus_samples: int = 3
    near_center_consensus_span_seconds: float = 0.30
    near_center_consensus_max_axis_spread_px: float = 1.25
    near_center_consensus_cooldown_seconds: float = 0.35
    final_reticle_enabled: bool = False
    final_reticle_track_anywhere: bool = True
    final_reticle_search_radius_px: int = 260
    final_reticle_trigger_distance_px: float = 5.0
    final_reticle_alignment_tolerance_px: float = 10.0
    runtime_profile_enabled: bool = True
    runtime_profile_pitch_reference_px_per_second: float = 14.0
    runtime_profile_yaw_reference_px_per_second: float = 12.0
    runtime_profile_learning_rate: float = 0.25
    runtime_profile_gain_scale_min: float = 0.55
    runtime_profile_gain_scale_max: float = 1.85
    missing_detection_fail_seconds: float = 1.50
    debug_window_enabled: bool = False
    debug_snapshot_interval_seconds: float = 5.0
    debug_window_scale: int = 4
    timeout_seconds: float | None = None
    filled_kp: float = 0.11
    filled_kd: float = 0.018
    hollow_kp: float = 0.035
    hollow_kd: float = 0.008
    pitch_engage_threshold: float = 0.22
    pitch_release_threshold: float = 0.08
    yaw_engage_threshold: float = 0.22
    yaw_release_threshold: float = 0.08
    controller_output_slew_per_second: float = 3.0
    error_derivative_low_pass_alpha: float = 0.35
    key_state_min_hold_seconds: float = 0.18
    key_state_min_release_seconds: float = 0.06
    oscillation_sign_flip_window_seconds: float = 1.0
    oscillation_sign_flip_threshold: int = 4
    fine_model_observation_floor_px: float = 0.35
    min_width: int = 1920
    min_height: int = 1080
    warm_h_min: int = 5
    warm_h_max: int = 45
    warm_s_min: int = 15
    warm_v_min: int = 110
    fallback_warm_s_min: int = 40
    fallback_warm_v_min: int = 100
    pale_h_max: int = 110
    pale_s_max: int = 90
    pale_v_min: int = 165
    pale_red_min: int = 130
    pale_green_min: int = 150
    pale_blue_red_delta_max: int = 50
    pale_green_red_delta_max: int = 60
    marker_min_area: int = 3
    marker_max_area: int = 80
    max_marker_distance_px: float = 80.0
    inner_disk_radius_px: int = 3
    outer_disk_radius_px: int = 6
    filled_inner_occupancy_threshold: float = 0.30
    definitive_filled_inner_occupancy_threshold: float = 0.80
    refinement_search_radius_px: int = 3
    capture_retry_attempts: int = 5
    capture_retry_interval_seconds: float = 0.10
    hollow_edge_phase_epsilon_degrees: float = 1.0
    hollow_edge_push_degrees: float = 6.0

    def roi_region(self) -> Region:
        half = self.roi_size // 2
        return (self.center_x - half, self.center_y - half, self.roi_size, self.roi_size)

    @property
    def outer_ring_radius_px(self) -> float:
        return self.compass_radius_px

    @property
    def inner_ring_radius_px(self) -> float:
        return max(1.0, self.compass_radius_px - self.inner_ring_outer_radius_offset_px)

    @property
    def compass_control_rotation_radians(self) -> float:
        return math.radians(self.compass_control_rotation_degrees)


@dataclass(slots=True)
class CompassMarker:
    marker_state: str
    marker_x: float
    marker_y: float
    dx: float
    dy: float
    control_dx: float
    control_dy: float
    distance: float
    normalized_radius: float
    front_semisphere_radians: float
    target_off_boresight_radians: float
    phase_adjustment_radians: float
    component_area: int
    inner_occupancy: float
    outer_ring_occupancy: float
    roi_region: Region
    compass_center_x: float | None = None
    compass_center_y: float | None = None
    compass_radius_estimate_px: float | None = None

    @property
    def is_filled(self) -> bool:
        return self.marker_state == "filled"

    @property
    def is_hollow(self) -> bool:
        return self.marker_state == "hollow"

    def is_aligned(self, config: AlignConfig) -> bool:
        return (
            self.is_filled
            and abs(self.dx) < config.axis_alignment_tolerance_px
            and abs(self.dy) < config.axis_alignment_tolerance_px
            and self.distance < config.alignment_tolerance_px
        )

    def is_confirmation_aligned(self, config: AlignConfig) -> bool:
        return (
            self.is_filled
            and abs(self.dx) < config.confirmation_axis_tolerance_px
            and abs(self.dy) < config.confirmation_axis_tolerance_px
            and self.distance < config.confirmation_distance_tolerance_px
        )

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "marker_state": self.marker_state,
            "marker_x": round(self.marker_x, 3),
            "marker_y": round(self.marker_y, 3),
            "relative_x": round(self.dx, 3),
            "relative_y": round(self.dy, 3),
            "screen_relative_x": round(self.dx, 3),
            "screen_relative_y": round(self.dy, 3),
            "control_relative_x": round(self.control_dx, 3),
            "control_relative_y": round(self.control_dy, 3),
            "distance": round(self.distance, 3),
            "normalized_radius": round(self.normalized_radius, 4),
            "front_semisphere_radians": round(self.front_semisphere_radians, 4),
            "front_semisphere_degrees": round(math.degrees(self.front_semisphere_radians), 3),
            "target_off_boresight_radians": round(self.target_off_boresight_radians, 4),
            "target_off_boresight_degrees": round(math.degrees(self.target_off_boresight_radians), 3),
            "phase_adjustment_radians": round(self.phase_adjustment_radians, 4),
            "phase_adjustment_degrees": round(math.degrees(self.phase_adjustment_radians), 3),
            "component_area": self.component_area,
            "inner_occupancy": round(self.inner_occupancy, 4),
            "outer_ring_occupancy": round(self.outer_ring_occupancy, 4),
            "roi_region": self.roi_region,
            "compass_center_x": None if self.compass_center_x is None else round(self.compass_center_x, 3),
            "compass_center_y": None if self.compass_center_y is None else round(self.compass_center_y, 3),
            "compass_radius_estimate_px": (
                None if self.compass_radius_estimate_px is None else round(self.compass_radius_estimate_px, 3)
            ),
        }


@dataclass(slots=True)
class CompassReadResult:
    status: str
    marker: CompassMarker | None = None

    @property
    def is_detected(self) -> bool:
        return self.status in {"filled", "hollow"} and self.marker is not None

    def to_debug_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.marker is not None:
            payload["marker"] = self.marker.to_debug_dict()
        return payload
