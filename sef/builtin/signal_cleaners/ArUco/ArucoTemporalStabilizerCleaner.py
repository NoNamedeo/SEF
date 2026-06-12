from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sef.core.artifacts.Signal import Signal
from sef.core.artifacts.signal_sample.ArucoMarkerSignalSample import (
    ArucoMarkerObservation,
    ArucoMarkerSignalSample,
    MarkerCorners,
)
from sef.core.interfaces.BufferContracts import IBuffer
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalSample import ISignalSample
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.interfaces.StreamingContracts import IStreamingSignalCleaner

Point2D = tuple[float, float]


@dataclass(slots=True)
class _MarkerFilterState:
    """Last filtered state retained per marker id."""

    center: Point2D
    corners: MarkerCorners | None


class ArucoTemporalStabilizerCleaner(IStreamingSignalCleaner):
    """Quality-aware temporal stabilizer for ArUco marker observations."""

    capabilities = StageCapabilities.streaming(
        stateful=True,
        preserves_order=True,
        realtime_safe=True,
    )

    DEFAULT_QUALITY_THRESHOLD = 0.45
    DEFAULT_ALPHA_HIGH_QUALITY = 0.65
    DEFAULT_ALPHA_LOW_QUALITY = 0.20
    DEFAULT_MAX_JUMP_PX = 2.0

    def __init__(
        self,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        alpha_high_quality: float = DEFAULT_ALPHA_HIGH_QUALITY,
        alpha_low_quality: float = DEFAULT_ALPHA_LOW_QUALITY,
        max_jump_px: float = DEFAULT_MAX_JUMP_PX,
        smooth_corners: bool = True,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.quality_threshold = float(quality_threshold)
        self.alpha_high_quality = float(alpha_high_quality)
        self.alpha_low_quality = float(alpha_low_quality)
        self.max_jump_px = float(max_jump_px)
        self.smooth_corners = bool(smooth_corners)

        if not (0.0 <= self.quality_threshold <= 1.0):
            raise ValueError("quality_threshold must be within [0, 1].")
        if not (0.0 <= self.alpha_low_quality <= 1.0):
            raise ValueError("alpha_low_quality must be within [0, 1].")
        if not (0.0 <= self.alpha_high_quality <= 1.0):
            raise ValueError("alpha_high_quality must be within [0, 1].")
        if self.alpha_low_quality > self.alpha_high_quality:
            raise ValueError("alpha_low_quality must be less than or equal to alpha_high_quality.")
        if self.max_jump_px < 0:
            raise ValueError("max_jump_px must be non-negative.")

    def clean(self, signal: ISignal) -> ISignal:
        samples = list(signal)
        if any(not isinstance(sample, ArucoMarkerSignalSample) for sample in samples):
            raise TypeError("ArucoTemporalStabilizerCleaner requires ArucoMarkerSignalSample inputs.")

        states: dict[int, _MarkerFilterState] = {}
        cleaned_samples = [self._clean_sample(sample, states) for sample in samples]
        return Signal(cleaned_samples, config=dict(signal.config))

    def clean_into(self, input_signal: Iterable[ISignalSample], output_buffer: IBuffer[ISignalSample]) -> None:
        states: dict[int, _MarkerFilterState] = {}
        try:
            for sample in input_signal:
                if not isinstance(sample, ArucoMarkerSignalSample):
                    raise TypeError("ArucoTemporalStabilizerCleaner requires ArucoMarkerSignalSample inputs.")
                output_buffer.put(self._clean_sample(sample, states))
        finally:
            output_buffer.close()

    def _clean_sample(
        self,
        sample: ArucoMarkerSignalSample,
        states: dict[int, _MarkerFilterState],
    ) -> ArucoMarkerSignalSample:
        cleaned_markers = [
            self._clean_observation(observation, states)
            for observation in sample.markers
        ]
        return ArucoMarkerSignalSample(
            frame_index=sample.frame_index,
            markers=cleaned_markers,
            timestamp_seconds=sample.timestamp_seconds,
            metadata={
                **dict(sample.metadata),
                "signal_cleaner": "aruco_temporal_stabilizer",
            },
        )

    def _clean_observation(
        self,
        observation: ArucoMarkerObservation,
        states: dict[int, _MarkerFilterState],
    ) -> ArucoMarkerObservation:
        state = states.get(observation.marker_id)
        if not observation.detected or observation.center is None:
            return self._copy_observation(
                observation,
                metadata_updates={
                    "temporal_stabilizer": {
                        "applied": False,
                        "reason": "not_detected",
                    }
                },
            )

        quality = self._normalized_quality(observation.quality_score)
        alpha = self._alpha_for_quality(quality)
        jump_px = math.dist(observation.center, state.center) if state is not None else 0.0
        gated_jump = bool(
            state is not None
            and self.max_jump_px > 0.0
            and jump_px > self.max_jump_px
            and quality < self.quality_threshold
        )
        if gated_jump:
            alpha = min(alpha, self.alpha_low_quality)

        filtered_corners = self._filter_corners(observation, state, alpha)
        filtered_center = self._filter_center(observation, state, alpha, filtered_corners)

        states[observation.marker_id] = _MarkerFilterState(
            center=filtered_center,
            corners=filtered_corners,
        )

        return self._copy_observation(
            observation,
            corners=filtered_corners,
            center=filtered_center,
            metadata_updates={
                "temporal_stabilizer": {
                    "applied": state is not None,
                    "quality": quality,
                    "alpha": alpha,
                    "jump_px": jump_px,
                    "gated_jump": gated_jump,
                    "smooth_corners": self.smooth_corners,
                }
            },
        )

    def _filter_corners(
        self,
        observation: ArucoMarkerObservation,
        state: _MarkerFilterState | None,
        alpha: float,
    ) -> MarkerCorners | None:
        if observation.corners is None:
            return None
        if not self.smooth_corners or state is None or state.corners is None:
            return observation.corners
        return tuple(
            self._ema_point(previous_point, current_point, alpha)
            for previous_point, current_point in zip(state.corners, observation.corners, strict=True)
        )  # type: ignore[return-value]

    def _filter_center(
        self,
        observation: ArucoMarkerObservation,
        state: _MarkerFilterState | None,
        alpha: float,
        filtered_corners: MarkerCorners | None,
    ) -> Point2D:
        if filtered_corners is not None:
            return self._corners_center(filtered_corners)
        if state is None:
            return observation.center
        return self._ema_point(state.center, observation.center, alpha)

    @staticmethod
    def _copy_observation(
        observation: ArucoMarkerObservation,
        *,
        corners: MarkerCorners | None | object = None,
        center: Point2D | None | object = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> ArucoMarkerObservation:
        resolved_corners = observation.corners if corners is None else corners
        resolved_center = observation.center if center is None else center
        metadata = dict(observation.metadata)
        if metadata_updates:
            metadata.update(metadata_updates)

        center_x = None
        center_y = None
        if resolved_center is not None:
            center_x = float(resolved_center[0])
            center_y = float(resolved_center[1])

        return ArucoMarkerObservation(
            marker_id=observation.marker_id,
            corners=resolved_corners,
            center_x=center_x,
            center_y=center_y,
            detected=observation.detected,
            quality_score=observation.quality_score,
            metadata=metadata,
        )

    def _alpha_for_quality(self, quality: float) -> float:
        return self.alpha_low_quality + ((self.alpha_high_quality - self.alpha_low_quality) * quality)

    @staticmethod
    def _normalized_quality(value: float | None) -> float:
        if value is None:
            return 0.0
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _ema_point(previous_point: Point2D, current_point: Point2D, alpha: float) -> Point2D:
        return (
            (alpha * current_point[0]) + ((1.0 - alpha) * previous_point[0]),
            (alpha * current_point[1]) + ((1.0 - alpha) * previous_point[1]),
        )

    @staticmethod
    def _corners_center(corners: MarkerCorners) -> Point2D:
        return (
            sum(point[0] for point in corners) / 4.0,
            sum(point[1] for point in corners) / 4.0,
        )
