from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from library.core.artifacts.COCOSkeletonSignalSample import (
    COCOSkeletonSignalSample,
)
from library.core.artifacts.Signal import Signal
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalSample import ISignalSample
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import (
    IStreamingSignalCleaner,
)


class COCOSkeletonNormalizationSignalCleaner(IStreamingSignalCleaner):
    """
    Streaming skeleton normalization cleaner.

    Operations:
        - pelvis centering
        - torso scale normalization
        - optional shoulder rotation alignment
    """

    capabilities = StageCapabilities.streaming(
        stateful=False,
        preserves_order=True,
        realtime_safe=True,
    )

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12

    def __init__(
        self,
        *,
        center_on_pelvis: bool = True,
        normalize_scale: bool = True,
        align_rotation: bool = False,
        min_scale: float = 1e-6,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        self.center_on_pelvis = bool(center_on_pelvis)
        self.normalize_scale = bool(normalize_scale)
        self.align_rotation = bool(align_rotation)
        self.min_scale = float(min_scale)

        if self.min_scale <= 0:
            raise ValueError("min_scale must be > 0.")

    def clean(self, signal: ISignal) -> ISignal:
        samples = list(signal)

        if any(not isinstance(sample, COCOSkeletonSignalSample) for sample in samples):
            raise TypeError(
                "COCOSkeletonNormalizationCleaner requires "
                "COCOSkeletonSignalSample inputs."
            )

        cleaned_samples = [
            self._clean_sample(sample)
            for sample in samples
        ]

        return Signal(
            cleaned_samples,
            config=dict(signal.config),
        )

    def clean_into(
        self,
        input_signal: Iterable[ISignalSample],
        output_buffer: SignalBuffer,
    ) -> None:
        try:
            for sample in input_signal:

                if not isinstance(sample, COCOSkeletonSignalSample):
                    raise TypeError(
                        "COCOSkeletonNormalizationCleaner requires "
                        "COCOSkeletonSignalSample inputs."
                    )

                output_buffer.put(self._clean_sample(sample))

        finally:
            output_buffer.close()

    def _clean_sample(
        self,
        sample: COCOSkeletonSignalSample,
    ) -> COCOSkeletonSignalSample:

        skeleton = np.asarray(sample.skeleton, dtype=np.float32).copy()

        metadata_updates: dict[str, Any] = {
            "skeleton_normalization": {
                "center_on_pelvis": self.center_on_pelvis,
                "normalize_scale": self.normalize_scale,
                "align_rotation": self.align_rotation,
            }
        }

        #centra lo skeleton nel "pelvis_center"

        pelvis_center = self._pelvis_center(skeleton)

        if pelvis_center is not None and self.center_on_pelvis:
            skeleton -= pelvis_center

        # riscala le dimensioni in base al torso

        torso_scale = self._torso_scale(skeleton)

        if (
            torso_scale is not None
            and torso_scale > self.min_scale
            and self.normalize_scale
        ):
            skeleton /= torso_scale

        metadata_updates["skeleton_normalization"]["torso_scale"] = torso_scale

        #ruota lo skeleton per metterlo "dritto"

        rotation_angle = None

        if self.align_rotation:
            rotation_angle = self._rotation_angle(skeleton)

            if rotation_angle is not None:
                skeleton = self._rotate_skeleton(
                    skeleton,
                    -rotation_angle,
                )

        metadata_updates["skeleton_normalization"]["rotation_angle_rad"] = (
            rotation_angle
        )

        centroid = self._compute_centroid(skeleton)

        return COCOSkeletonSignalSample(
            frame_index=sample.frame_index,
            skeleton=skeleton,
            confidence=np.asarray(sample.confidence).copy(),
            centroid=centroid,
            timestamp_seconds=sample.timestamp_seconds,
            metadata={
                **dict(sample.metadata),
                **metadata_updates,
            },
        )

    def _pelvis_center(self, skeleton: np.ndarray) -> np.ndarray | None:
        left_hip = skeleton[self.LEFT_HIP]
        right_hip = skeleton[self.RIGHT_HIP]

        if self._invalid_point(left_hip) or self._invalid_point(right_hip):
            return None

        return (left_hip + right_hip) / 2.0

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

        if (
            self._invalid_point(left_shoulder)
            or self._invalid_point(right_shoulder)
        ):
            return None

        delta = right_shoulder - left_shoulder

        return float(np.arctan2(delta[1], delta[0]))

    @staticmethod
    def _rotate_skeleton(
        skeleton: np.ndarray,
        angle_rad: float,
    ) -> np.ndarray:

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
    def _compute_centroid(skeleton: np.ndarray) -> tuple[float, float]:
        valid = skeleton[np.any(skeleton != 0, axis=1)]

        if len(valid) == 0:
            return (0.0, 0.0)

        centroid = valid.mean(axis=0)

        return (
            float(centroid[0]),
            float(centroid[1]),
        )

    @staticmethod
    def _invalid_point(point: np.ndarray) -> bool:
        return bool(np.all(point == 0))