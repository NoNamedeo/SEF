from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class COCOSkeletonNormalizationConfig:
    """Configuration for COCO 17-keypoint skeleton normalization."""

    center_on_pelvis: bool = True
    normalize_scale: bool = True
    align_rotation: bool = False
    min_scale: float = 1e-6

    def __post_init__(self) -> None:
        if self.min_scale <= 0:
            raise ValueError("min_scale must be greater than 0.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> COCOSkeletonNormalizationConfig:
        """Build normalization config from optional user/runtime configuration."""
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("skeleton_normalization config must be a mapping.")
        return cls(
            center_on_pelvis=bool(value.get("center_on_pelvis", True)),
            normalize_scale=bool(value.get("normalize_scale", True)),
            align_rotation=bool(value.get("align_rotation", False)),
            min_scale=float(value.get("min_scale", 1e-6)),
        )


@dataclass(frozen=True, slots=True)
class COCOSkeletonNormalizationResult:
    """Normalized skeleton plus metadata needed to inspect the transform."""

    skeleton: np.ndarray
    pelvis_center: tuple[float, float] | None
    torso_scale: float | None
    rotation_angle_rad: float | None
    config: COCOSkeletonNormalizationConfig

    def metadata(self) -> dict[str, Any]:
        """Return serializable metadata for model/debug observability."""
        return {
            "center_on_pelvis": self.config.center_on_pelvis,
            "normalize_scale": self.config.normalize_scale,
            "align_rotation": self.config.align_rotation,
            "pelvis_center": self.pelvis_center,
            "torso_scale": self.torso_scale,
            "rotation_angle_rad": self.rotation_angle_rad,
        }


class COCOSkeletonNormalizer:
    """
    Normalize COCO 17-keypoint skeletons for movement classification.

    The realtime visualizer must keep raw pixel coordinates. This normalizer is
    intended for model features only, matching the preprocessing used when the
    tennis classifier dataset is generated.
    """

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12

    def __init__(self, config: COCOSkeletonNormalizationConfig | None = None) -> None:
        self.config = config or COCOSkeletonNormalizationConfig()

    def normalize(self, skeleton: np.ndarray | list) -> COCOSkeletonNormalizationResult:
        """Return a normalized skeleton copy without mutating ``skeleton``."""
        normalized = np.asarray(skeleton, dtype=np.float32).copy()

        pelvis_center = self._pelvis_center(normalized)
        if pelvis_center is not None and self.config.center_on_pelvis:
            normalized -= np.asarray(pelvis_center, dtype=np.float32)

        torso_scale = self._torso_scale(normalized)
        if (
            torso_scale is not None
            and torso_scale > self.config.min_scale
            and self.config.normalize_scale
        ):
            normalized /= torso_scale

        rotation_angle = None
        if self.config.align_rotation:
            rotation_angle = self._rotation_angle(normalized)
            if rotation_angle is not None:
                normalized = self._rotate_skeleton(normalized, -rotation_angle)

        return COCOSkeletonNormalizationResult(
            skeleton=normalized,
            pelvis_center=pelvis_center,
            torso_scale=torso_scale,
            rotation_angle_rad=rotation_angle,
            config=self.config,
        )

    @staticmethod
    def compute_centroid(skeleton: np.ndarray) -> tuple[float, float]:
        """Compute centroid over valid non-zero keypoints."""
        valid_points = skeleton[np.any(skeleton != 0, axis=1)]
        if len(valid_points) == 0:
            return 0.0, 0.0
        centroid = valid_points.mean(axis=0)
        return float(centroid[0]), float(centroid[1])

    def _pelvis_center(self, skeleton: np.ndarray) -> tuple[float, float] | None:
        left_hip = skeleton[self.LEFT_HIP]
        right_hip = skeleton[self.RIGHT_HIP]
        if self._invalid_point(left_hip) or self._invalid_point(right_hip):
            return None
        center = (left_hip + right_hip) / 2.0
        return float(center[0]), float(center[1])

    def _torso_scale(self, skeleton: np.ndarray) -> float | None:
        left_shoulder = skeleton[self.LEFT_SHOULDER]
        right_shoulder = skeleton[self.RIGHT_SHOULDER]
        left_hip = skeleton[self.LEFT_HIP]
        right_hip = skeleton[self.RIGHT_HIP]

        if (
            self._invalid_point(left_shoulder)
            or self._invalid_point(right_shoulder)
            or self._invalid_point(left_hip)
            or self._invalid_point(right_hip)
        ):
            return None

        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        hip_center = (left_hip + right_hip) / 2.0
        return float(np.linalg.norm(shoulder_center - hip_center))

    def _rotation_angle(self, skeleton: np.ndarray) -> float | None:
        left_shoulder = skeleton[self.LEFT_SHOULDER]
        right_shoulder = skeleton[self.RIGHT_SHOULDER]
        if self._invalid_point(left_shoulder) or self._invalid_point(right_shoulder):
            return None
        delta = right_shoulder - left_shoulder
        return float(np.arctan2(delta[1], delta[0]))

    @staticmethod
    def _rotate_skeleton(skeleton: np.ndarray, angle_rad: float) -> np.ndarray:
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        rotation_matrix = np.array(
            [
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta],
            ],
            dtype=np.float32,
        )
        return skeleton @ rotation_matrix.T

    @staticmethod
    def _invalid_point(point: np.ndarray) -> bool:
        return bool(np.all(point == 0))
