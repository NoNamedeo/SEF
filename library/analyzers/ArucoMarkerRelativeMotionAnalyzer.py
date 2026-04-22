from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import Any

from library.core.artifacts.ArucoMarkerSignalSample import ArucoMarkerSignalSample
from library.core.artifacts.ArucoRelativeMotionData import (
    ArucoMarkerRelativeMotionData,
    ArucoMarkerRelativeMotionSeries,
    MarkerPair,
)
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.ISignal import ISignal


class ArucoMarkerRelativeMotionAnalyzer(IAnalyzer):
    """Measure time-varying distances between marker pairs."""

    DEFAULT_TITLE = "ArUco Relative Motion"

    def __init__(
        self,
        marker_pairs: Sequence[Sequence[int]] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.marker_pairs = self._normalize_pairs(marker_pairs)
        self.use_timestamps = bool(self.config.get("use_timestamps", True))
        self._title = str(self.config.get("title", self.DEFAULT_TITLE))

    def analyze(self, signal: ISignal) -> ArucoMarkerRelativeMotionData:
        samples = self._aruco_samples(signal)
        marker_pairs = self.marker_pairs or self._auto_pairs(samples)

        series: list[ArucoMarkerRelativeMotionSeries] = []
        for marker_pair in marker_pairs:
            baseline_distance = self._baseline_distance(samples, marker_pair)
            if baseline_distance is None:
                continue
            series.append(self._build_series(samples, marker_pair, baseline_distance))

        if not series:
            raise ValueError("ArucoMarkerRelativeMotionAnalyzer requires at least one marker pair with detections.")

        return ArucoMarkerRelativeMotionData(
            series=series,
            title=self._title,
            metadata={
                "title": self._title,
                "marker_pairs": [list(pair) for pair in marker_pairs],
                "use_timestamps": self.use_timestamps,
            },
        )

    def _build_series(
        self,
        samples: list[ArucoMarkerSignalSample],
        marker_pair: MarkerPair,
        baseline_distance: float,
    ) -> ArucoMarkerRelativeMotionSeries:
        frame_indices: list[int] = []
        timestamps: list[float | None] = []
        detected_flags: list[bool] = []
        distances: list[float] = []
        deltas: list[float] = []

        first_marker_id, second_marker_id = marker_pair
        for sample in samples:
            first_marker = sample.marker_by_id(first_marker_id)
            second_marker = sample.marker_by_id(second_marker_id)
            frame_indices.append(int(sample.frame_index))
            timestamps.append(sample.timestamp_seconds)

            if (
                first_marker is None
                or second_marker is None
                or not first_marker.detected
                or not second_marker.detected
                or first_marker.center is None
                or second_marker.center is None
            ):
                detected_flags.append(False)
                distances.append(float("nan"))
                deltas.append(float("nan"))
                continue

            current_distance = math.dist(first_marker.center, second_marker.center)
            detected_flags.append(True)
            distances.append(current_distance)
            deltas.append(current_distance - baseline_distance)

        return ArucoMarkerRelativeMotionSeries(
            marker_pair=marker_pair,
            frame_indices=frame_indices,
            timestamps=timestamps,
            detected=detected_flags,
            distances=distances,
            distance_deltas=deltas,
            baseline_distance=baseline_distance,
            stats=self._stats(distances, deltas, detected_flags),
            metadata={"use_timestamps": self.use_timestamps},
        )

    @staticmethod
    def _aruco_samples(signal: ISignal) -> list[ArucoMarkerSignalSample]:
        samples = list(signal)
        if any(not isinstance(sample, ArucoMarkerSignalSample) for sample in samples):
            raise TypeError("ArucoMarkerRelativeMotionAnalyzer requires ArucoMarkerSignalSample inputs.")
        return samples

    @staticmethod
    def _normalize_pairs(
        marker_pairs: Sequence[Sequence[int]] | None,
    ) -> tuple[MarkerPair, ...] | None:
        if marker_pairs is None:
            return None
        normalized_pairs: set[MarkerPair] = set()
        for pair in marker_pairs:
            if len(pair) != 2:
                raise ValueError("Each marker pair must contain exactly two marker ids.")
            normalized_pairs.add(tuple(sorted((int(pair[0]), int(pair[1])))))
        return tuple(sorted(normalized_pairs))

    @staticmethod
    def _auto_pairs(samples: list[ArucoMarkerSignalSample]) -> tuple[MarkerPair, ...]:
        marker_ids = sorted(
            {
                marker.marker_id
                for sample in samples
                for marker in sample.markers
                if marker.detected
            }
        )
        return tuple((int(first_id), int(second_id)) for first_id, second_id in combinations(marker_ids, 2))

    @staticmethod
    def _baseline_distance(
        samples: list[ArucoMarkerSignalSample],
        marker_pair: MarkerPair,
    ) -> float | None:
        first_marker_id, second_marker_id = marker_pair
        for sample in samples:
            first_marker = sample.marker_by_id(first_marker_id)
            second_marker = sample.marker_by_id(second_marker_id)
            if (
                first_marker is not None
                and second_marker is not None
                and first_marker.detected
                and second_marker.detected
                and first_marker.center is not None
                and second_marker.center is not None
            ):
                return math.dist(first_marker.center, second_marker.center)
        return None

    @staticmethod
    def _stats(
        distances: list[float],
        deltas: list[float],
        detected_flags: list[bool],
    ) -> dict[str, float]:
        finite_distances = [value for value in distances if math.isfinite(value)]
        finite_deltas = [value for value in deltas if math.isfinite(value)]
        return {
            "detected_samples": float(sum(detected_flags)),
            "mean_distance": (sum(finite_distances) / len(finite_distances)) if finite_distances else 0.0,
            "max_abs_delta": max((abs(value) for value in finite_deltas), default=0.0),
        }
