from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from library.core.artifacts.ArucoMarkerSignalSample import (
    ArucoMarkerObservation,
    ArucoMarkerSignalSample,
    MarkerCorners,
)
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor


@dataclass(frozen=True, slots=True)
class _CornerRefinementConfig:
    """Subpixel corner refinement options."""

    enabled: bool
    method: str
    win_size: int
    max_iterations: int
    min_accuracy: float


@dataclass(frozen=True, slots=True)
class _PreprocessingConfig:
    """Optional pre-detection image preprocessing options."""

    gaussian_blur_enabled: bool
    gaussian_blur_kernel_size: int
    clahe_enabled: bool
    clahe_clip_limit: float
    clahe_tile_grid_size: tuple[int, int]
    denoise_enabled: bool
    denoise_h: float


@dataclass(frozen=True, slots=True)
class _PoseEstimationConfig:
    """Future-facing pose estimation placeholders."""

    estimate_pose: bool
    marker_length: float | None
    camera_matrix: np.ndarray | None
    dist_coeffs: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _DetectionBatch:
    """Raw detector output enriched with extractor-level metadata."""

    corners: tuple[np.ndarray, ...]
    ids: np.ndarray | None
    rejected: tuple[np.ndarray, ...]
    used_padding: bool
    refinement_applied: bool
    refinement_method: str


class ArucoMarkerSignalExtractor(ISignalExtractor):
    """Detect DICT_6X6_250 ArUco markers frame by frame."""

    DICTIONARY_ID = cv2.aruco.DICT_6X6_250
    DEFAULT_WHITE_BORDER_PADDING_PX = 32

    QUALITY_MODEL = "aruco_area_border_shape_v1"
    QUALITY_AREA_REFERENCE_RATIO = 0.0025
    QUALITY_BORDER_REFERENCE_RATIO = 0.03
    QUALITY_AREA_WEIGHT = 0.45
    QUALITY_BORDER_WEIGHT = 0.20
    QUALITY_SHAPE_WEIGHT = 0.35

    def __init__(
        self,
        marker_ids: Sequence[int] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.marker_ids = tuple(sorted({int(marker_id) for marker_id in marker_ids})) if marker_ids is not None else None
        self._white_border_padding_px = int(self.config.get("white_border_padding_px", self.DEFAULT_WHITE_BORDER_PADDING_PX))
        if self._white_border_padding_px < 0:
            raise ValueError("white_border_padding_px must be non-negative")

        self._corner_refinement = self._build_corner_refinement_config()
        self._preprocessing = self._build_preprocessing_config()
        self._pose_estimation = self._build_pose_estimation_config()
        self._validate_future_pose_config()

        self._dictionary = cv2.aruco.getPredefinedDictionary(self.DICTIONARY_ID)
        self._detector_parameters = self._build_detector_parameters()
        self._builtin_corner_refinement = self._configure_builtin_corner_refinement(self._detector_parameters)
        self._manual_corner_refinement = self._should_use_manual_corner_refinement()
        self._detector = self._build_detector()

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples: list[ArucoMarkerSignalSample] = []
        known_marker_ids = set(self.marker_ids or ())

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position
            detection_frame = self._prepare_detection_frame(frame.frame)
            detection = self._detect_with_white_border_fallback(detection_frame)

            detected_observations = self._build_detected_observations(
                detection=detection,
                frame_shape=detection_frame.shape,
            )

            known_marker_ids.update(detected_observations.keys())
            observations = self._build_frame_observations(
                detected_observations=detected_observations,
                known_marker_ids=known_marker_ids,
            )

            samples.append(
                ArucoMarkerSignalSample(
                    frame_index=int(frame_index),
                    markers=observations,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata={
                        **dict(frame.metadata),
                        "detected_marker_count": sum(1 for observation in observations if observation.detected),
                        "rejected_candidate_count": len(detection.rejected),
                        "white_border_fallback_used": detection.used_padding,
                        "corner_refinement_applied": detection.refinement_applied,
                        "corner_refinement_method": detection.refinement_method,
                    },
                )
            )

        return Signal(samples)

    def _build_corner_refinement_config(self) -> _CornerRefinementConfig:
        enabled = bool(self.config.get("corner_refinement_enabled", False))
        method = str(self.config.get("corner_refinement_method", "subpix")).strip().lower()
        win_size = int(self.config.get("corner_refinement_win_size", 5))
        max_iterations = int(self.config.get("corner_refinement_max_iterations", 50))
        min_accuracy = float(self.config.get("corner_refinement_min_accuracy", 0.01))

        if win_size <= 0:
            raise ValueError("corner_refinement_win_size must be greater than 0")
        if max_iterations <= 0:
            raise ValueError("corner_refinement_max_iterations must be greater than 0")
        if min_accuracy <= 0:
            raise ValueError("corner_refinement_min_accuracy must be greater than 0")

        normalized_method = self._normalize_corner_refinement_method(method)
        return _CornerRefinementConfig(
            enabled=enabled,
            method=normalized_method,
            win_size=win_size,
            max_iterations=max_iterations,
            min_accuracy=min_accuracy,
        )

    def _build_preprocessing_config(self) -> _PreprocessingConfig:
        blur_kernel_size = int(self.config.get("gaussian_blur_kernel_size", 3))
        if blur_kernel_size <= 0:
            raise ValueError("gaussian_blur_kernel_size must be greater than 0")
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1

        clahe_grid_size = self._normalize_size_pair(
            self.config.get("clahe_tile_grid_size", (8, 8)),
            config_key="clahe_tile_grid_size",
        )
        denoise_h = float(self.config.get("denoise_h", 3.0))
        if denoise_h < 0:
            raise ValueError("denoise_h must be non-negative")

        return _PreprocessingConfig(
            gaussian_blur_enabled=bool(self.config.get("gaussian_blur_enabled", False)),
            gaussian_blur_kernel_size=blur_kernel_size,
            clahe_enabled=bool(self.config.get("clahe_enabled", False)),
            clahe_clip_limit=float(self.config.get("clahe_clip_limit", 2.0)),
            clahe_tile_grid_size=clahe_grid_size,
            denoise_enabled=bool(self.config.get("denoise_enabled", False)),
            denoise_h=denoise_h,
        )

    def _build_pose_estimation_config(self) -> _PoseEstimationConfig:
        marker_length = self.config.get("marker_length")
        if marker_length is not None:
            marker_length = float(marker_length)
            if marker_length <= 0:
                raise ValueError("marker_length must be greater than 0")

        camera_matrix = self._normalize_matrix(self.config.get("camera_matrix"), config_key="camera_matrix")
        dist_coeffs = self._normalize_vector(self.config.get("dist_coeffs"), config_key="dist_coeffs")

        return _PoseEstimationConfig(
            estimate_pose=bool(self.config.get("estimate_pose", False)),
            marker_length=marker_length,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )

    def _validate_future_pose_config(self) -> None:
        if not self._pose_estimation.estimate_pose:
            return
        raise NotImplementedError("Pose estimation hooks are configured but not implemented yet for ArucoMarkerSignalExtractor.")

    def _build_detector_parameters(self):
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()

        self._apply_detector_parameter_overrides(parameters)
        return parameters

    def _apply_detector_parameter_overrides(self, parameters: Any) -> None:
        detector_parameter_specs: tuple[tuple[str, str, type], ...] = (
            ("adaptive_thresh_win_size_min", "adaptiveThreshWinSizeMin", int),
            ("adaptive_thresh_win_size_max", "adaptiveThreshWinSizeMax", int),
            ("adaptive_thresh_win_size_step", "adaptiveThreshWinSizeStep", int),
            ("adaptive_thresh_constant", "adaptiveThreshConstant", float),
            ("min_marker_perimeter_rate", "minMarkerPerimeterRate", float),
            ("max_marker_perimeter_rate", "maxMarkerPerimeterRate", float),
            ("polygonal_approx_accuracy_rate", "polygonalApproxAccuracyRate", float),
            ("min_corner_distance_rate", "minCornerDistanceRate", float),
            ("min_distance_to_border", "minDistanceToBorder", int),
            ("min_marker_distance_rate", "minMarkerDistanceRate", float),
            ("perspective_remove_pixel_per_cell", "perspectiveRemovePixelPerCell", int),
            ("perspective_remove_ignored_margin_per_cell", "perspectiveRemoveIgnoredMarginPerCell", float),
            ("error_correction_rate", "errorCorrectionRate", float),
        )

        for config_key, parameter_name, cast in detector_parameter_specs:
            if config_key not in self.config or not hasattr(parameters, parameter_name):
                continue
            setattr(parameters, parameter_name, cast(self.config[config_key]))

    def _configure_builtin_corner_refinement(self, parameters: Any) -> str:
        if not self._corner_refinement.enabled or self._corner_refinement.method == "none":
            return "none"

        if not hasattr(parameters, "cornerRefinementMethod"):
            return "none"

        refinement_mapping = {
            "none": getattr(cv2.aruco, "CORNER_REFINE_NONE", None),
            "subpix": getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", None),
            "contour": getattr(cv2.aruco, "CORNER_REFINE_CONTOUR", None),
            "apriltag": getattr(cv2.aruco, "CORNER_REFINE_APRILTAG", None),
        }
        if self._corner_refinement.method not in refinement_mapping:
            return "none"

        builtin_method = refinement_mapping[self._corner_refinement.method]
        if builtin_method is None:
            return "none"

        parameters.cornerRefinementMethod = builtin_method
        if hasattr(parameters, "cornerRefinementWinSize"):
            parameters.cornerRefinementWinSize = self._corner_refinement.win_size
        if hasattr(parameters, "cornerRefinementMaxIterations"):
            parameters.cornerRefinementMaxIterations = self._corner_refinement.max_iterations
        if hasattr(parameters, "cornerRefinementMinAccuracy"):
            parameters.cornerRefinementMinAccuracy = self._corner_refinement.min_accuracy
        return self._corner_refinement.method

    def _should_use_manual_corner_refinement(self) -> bool:
        if not self._corner_refinement.enabled:
            return False
        return self._corner_refinement.method == "manual_subpix" or (
            self._corner_refinement.method == "subpix" and self._builtin_corner_refinement == "none"
        )

    def _build_detector(self):
        if hasattr(cv2.aruco, "ArucoDetector"):
            return cv2.aruco.ArucoDetector(self._dictionary, self._detector_parameters)
        return None

    def _prepare_detection_frame(self, frame: np.ndarray) -> np.ndarray:
        grayscale = self._to_grayscale(frame)
        return self._apply_optional_preprocessing(grayscale)

    def _apply_optional_preprocessing(self, grayscale_frame: np.ndarray) -> np.ndarray:
        processed = grayscale_frame

        if self._preprocessing.gaussian_blur_enabled:
            processed = cv2.GaussianBlur(
                processed,
                (self._preprocessing.gaussian_blur_kernel_size, self._preprocessing.gaussian_blur_kernel_size),
                0,
            )
        if self._preprocessing.clahe_enabled:
            clahe = cv2.createCLAHE(
                clipLimit=self._preprocessing.clahe_clip_limit,
                tileGridSize=self._preprocessing.clahe_tile_grid_size,
            )
            processed = clahe.apply(processed)
        if self._preprocessing.denoise_enabled:
            processed = cv2.fastNlMeansDenoising(processed, None, h=self._preprocessing.denoise_h)

        return processed

    def _detect_with_white_border_fallback(self, grayscale_frame: np.ndarray) -> _DetectionBatch:
        primary_detection = self._detect_once(grayscale_frame, used_padding=False)
        if self._has_detections(primary_detection) or self._white_border_padding_px == 0:
            return primary_detection

        padded_frame = self._pad_frame(grayscale_frame)
        padded_detection = self._detect_once(padded_frame, used_padding=True)
        if not self._has_detections(padded_detection):
            if primary_detection.rejected:
                return primary_detection
            return _DetectionBatch(
                corners=primary_detection.corners,
                ids=primary_detection.ids,
                rejected=padded_detection.rejected,
                used_padding=False,
                refinement_applied=primary_detection.refinement_applied,
                refinement_method=primary_detection.refinement_method,
            )

        return self._adjust_detection_from_padded_frame(padded_detection)

    def _detect_once(self, grayscale_frame: np.ndarray, *, used_padding: bool) -> _DetectionBatch:
        corners, ids, rejected = self._detect_markers_raw(grayscale_frame)
        refined_corners, refinement_applied, refinement_method = self._refine_detected_corners(
            grayscale_frame,
            list(corners or []),
        )
        return _DetectionBatch(
            corners=tuple(refined_corners),
            ids=ids,
            rejected=tuple(rejected or []),
            used_padding=used_padding,
            refinement_applied=refinement_applied,
            refinement_method=refinement_method,
        )

    def _refine_detected_corners(
        self,
        grayscale_frame: np.ndarray,
        corners: list[np.ndarray],
    ) -> tuple[list[np.ndarray], bool, str]:
        if not corners:
            return corners, False, "none"

        if self._manual_corner_refinement:
            return (
                self._manual_subpixel_refinement(grayscale_frame, corners),
                True,
                "manual_subpix",
            )

        if self._builtin_corner_refinement != "none":
            return corners, True, self._builtin_corner_refinement

        return corners, False, "none"

    def _manual_subpixel_refinement(
        self,
        grayscale_frame: np.ndarray,
        corners: list[np.ndarray],
    ) -> list[np.ndarray]:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self._corner_refinement.max_iterations,
            self._corner_refinement.min_accuracy,
        )
        refined_corners: list[np.ndarray] = []

        for raw_corners in corners:
            points = np.asarray(raw_corners, dtype=np.float32).reshape(-1, 1, 2)
            refined_points = cv2.cornerSubPix(
                grayscale_frame,
                points,
                (self._corner_refinement.win_size, self._corner_refinement.win_size),
                (-1, -1),
                criteria,
            )
            refined_corners.append(np.asarray(refined_points, dtype=np.float32).reshape(1, 4, 2))

        return refined_corners

    def _detect_markers_raw(
        self,
        grayscale_frame: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
        if self._detector is not None:
            corners, ids, rejected = self._detector.detectMarkers(grayscale_frame)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                grayscale_frame,
                self._dictionary,
                parameters=self._detector_parameters,
            )
        return list(corners or []), ids, list(rejected or [])

    def _adjust_detection_from_padded_frame(self, detection: _DetectionBatch) -> _DetectionBatch:
        padding_offset = np.array(
            [[[self._white_border_padding_px, self._white_border_padding_px]]],
            dtype=np.float32,
        )
        adjusted_corners = tuple(np.asarray(raw_corners, dtype=np.float32) - padding_offset for raw_corners in detection.corners)
        return _DetectionBatch(
            corners=adjusted_corners,
            ids=detection.ids,
            rejected=detection.rejected,
            used_padding=True,
            refinement_applied=detection.refinement_applied,
            refinement_method=detection.refinement_method,
        )

    def _build_frame_observations(
        self,
        *,
        detected_observations: dict[int, ArucoMarkerObservation],
        known_marker_ids: set[int],
    ) -> list[ArucoMarkerObservation]:
        if self.marker_ids is not None:
            frame_marker_ids = self.marker_ids
        elif known_marker_ids:
            frame_marker_ids = tuple(sorted(known_marker_ids))
        else:
            frame_marker_ids = tuple(sorted(detected_observations))

        observations: list[ArucoMarkerObservation] = []
        for marker_id in frame_marker_ids:
            observation = detected_observations.get(marker_id)
            if observation is None:
                observations.append(
                    ArucoMarkerObservation(
                        marker_id=marker_id,
                        corners=None,
                        center_x=None,
                        center_y=None,
                        detected=False,
                        quality_score=None,
                        metadata={"reason": "not_detected"},
                    )
                )
                continue
            observations.append(observation)
        return observations

    def _build_detected_observations(
        self,
        *,
        detection: _DetectionBatch,
        frame_shape: tuple[int, int],
    ) -> dict[int, ArucoMarkerObservation]:
        if detection.ids is None or len(detection.corners) == 0:
            return {}

        observations: dict[int, ArucoMarkerObservation] = {}
        for marker_id, raw_corners in zip(detection.ids.flatten().tolist(), detection.corners):
            observation = self._build_detected_observation(
                marker_id=int(marker_id),
                raw_corners=raw_corners,
                frame_shape=frame_shape,
                detection=detection,
            )
            observations[observation.marker_id] = observation
        return observations

    def _build_detected_observation(
        self,
        *,
        marker_id: int,
        raw_corners: np.ndarray,
        frame_shape: tuple[int, int],
        detection: _DetectionBatch,
    ) -> ArucoMarkerObservation:
        normalized_corners = self._normalize_corners(raw_corners)
        area_px = self._compute_area_px(normalized_corners)
        center_x, center_y = self._compute_center(normalized_corners)
        quality_components = self._compute_quality_components(
            corners=normalized_corners,
            frame_shape=frame_shape,
            area_px=area_px,
        )
        quality_score = self._compose_quality_score(quality_components)

        metadata = self._build_observation_metadata(
            area_px=area_px,
            detection=detection,
            quality_components=quality_components,
        )
        metadata.update(self._build_pose_extension_metadata())

        return ArucoMarkerObservation(
            marker_id=marker_id,
            corners=normalized_corners,
            center_x=center_x,
            center_y=center_y,
            detected=True,
            quality_score=quality_score,
            metadata=metadata,
        )

    @staticmethod
    def _compute_center(corners: MarkerCorners) -> tuple[float, float]:
        center_x = sum(point[0] for point in corners) / 4.0
        center_y = sum(point[1] for point in corners) / 4.0
        return center_x, center_y

    @staticmethod
    def _compute_area_px(corners: MarkerCorners) -> float:
        contour = np.asarray(corners, dtype=np.float32)
        return float(abs(cv2.contourArea(contour)))

    def _compute_quality_components(
        self,
        *,
        corners: MarkerCorners,
        frame_shape: tuple[int, int],
        area_px: float,
    ) -> dict[str, float]:
        frame_height, frame_width = frame_shape[:2]
        frame_area = float(frame_height * frame_width) if frame_height > 0 and frame_width > 0 else 0.0
        normalized_area = (area_px / frame_area) if frame_area > 0 else 0.0
        area_score = self._score_marker_area(normalized_area)

        border_margin_px = self._minimum_border_margin_px(corners, frame_width, frame_height)
        border_margin_normalized = border_margin_px / float(min(frame_width, frame_height)) if frame_width > 0 and frame_height > 0 else 0.0
        border_score = self._score_border_distance(border_margin_normalized)

        shape_score = self._score_shape_regularity(corners, area_px)

        return {
            "normalized_area": normalized_area,
            "area_score": area_score,
            "border_margin_px": border_margin_px,
            "border_margin_normalized": border_margin_normalized,
            "border_score": border_score,
            "shape_score": shape_score,
        }

    def _score_marker_area(self, normalized_area: float) -> float:
        return self._clamp(normalized_area / self.QUALITY_AREA_REFERENCE_RATIO)

    def _score_border_distance(self, normalized_border_margin: float) -> float:
        return self._clamp(normalized_border_margin / self.QUALITY_BORDER_REFERENCE_RATIO)

    def _score_shape_regularity(self, corners: MarkerCorners, area_px: float) -> float:
        contour = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0 or area_px <= 0:
            return 0.0

        compactness = (4.0 * math.pi * area_px) / (perimeter * perimeter)
        ideal_square_compactness = math.pi / 4.0
        compactness_score = self._clamp(compactness / ideal_square_compactness)
        convexity_score = 1.0 if cv2.isContourConvex(contour) else 0.0
        return (0.8 * compactness_score) + (0.2 * convexity_score)

    def _compose_quality_score(self, components: dict[str, float]) -> float:
        return (
            components["area_score"] * self.QUALITY_AREA_WEIGHT
            + components["border_score"] * self.QUALITY_BORDER_WEIGHT
            + components["shape_score"] * self.QUALITY_SHAPE_WEIGHT
        )

    def _build_observation_metadata(
        self,
        *,
        area_px: float,
        detection: _DetectionBatch,
        quality_components: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "area_px": area_px,
            "quality_model": self.QUALITY_MODEL,
            "quality_components": quality_components,
            "refinement_applied": detection.refinement_applied,
            "refinement_method": detection.refinement_method,
            "used_white_border_fallback": detection.used_padding,
        }

    def _build_pose_extension_metadata(self) -> dict[str, Any]:
        if not self._pose_estimation.estimate_pose:
            return {}
        return {"pose_estimation_requested": True}

    @staticmethod
    def _minimum_border_margin_px(
        corners: MarkerCorners,
        frame_width: int,
        frame_height: int,
    ) -> float:
        distances = (
            [point[0] for point in corners]
            + [point[1] for point in corners]
            + [float(frame_width - 1) - point[0] for point in corners]
            + [float(frame_height - 1) - point[1] for point in corners]
        )
        return max(0.0, min(distances, default=0.0))

    @staticmethod
    def _has_detections(detection: _DetectionBatch) -> bool:
        return detection.ids is not None and len(detection.corners) > 0

    def _pad_frame(self, grayscale_frame: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(
            grayscale_frame,
            self._white_border_padding_px,
            self._white_border_padding_px,
            self._white_border_padding_px,
            self._white_border_padding_px,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[:, :, 0]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _normalize_corners(raw_corners: np.ndarray) -> MarkerCorners:
        squeezed = np.asarray(raw_corners, dtype=np.float32).reshape(4, 2)
        return tuple((float(x), float(y)) for x, y in squeezed)  # type: ignore[return-value]

    @staticmethod
    def _normalize_corner_refinement_method(method: str) -> str:
        aliases = {
            "none": "none",
            "subpix": "subpix",
            "contour": "contour",
            "apriltag": "apriltag",
            "manual_subpix": "manual_subpix",
            "manual": "manual_subpix",
        }
        normalized_method = aliases.get(method)
        if normalized_method is None:
            supported = ", ".join(sorted(aliases))
            raise ValueError(f"Unsupported corner_refinement_method '{method}'. Supported values: {supported}.")
        return normalized_method

    @staticmethod
    def _normalize_size_pair(value: Any, *, config_key: str) -> tuple[int, int]:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"{config_key} must contain exactly two integer values.")
        width, height = int(value[0]), int(value[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"{config_key} values must be greater than 0.")
        return width, height

    @staticmethod
    def _normalize_matrix(value: Any, *, config_key: str) -> np.ndarray | None:
        if value is None:
            return None
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"{config_key} must be a 3x3 matrix.")
        return matrix

    @staticmethod
    def _normalize_vector(value: Any, *, config_key: str) -> np.ndarray | None:
        if value is None:
            return None
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size == 0:
            raise ValueError(f"{config_key} must not be empty.")
        return vector

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))
