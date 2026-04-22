from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from library.core.artifacts.ArucoDisplacementData import (
    ArucoMarkerDisplacementData,
    ArucoMarkerDisplacementSeries,
)
from library.core.artifacts.ArucoMarkerSignalSample import ArucoMarkerSignalSample, Point2D
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.ISignal import ISignal


class ArucoMarkerDisplacementAnalyzer(IAnalyzer):
    """Compute 2D displacement over time for each detected marker."""

    DEFAULT_TITLE = "ArUco Marker Displacement"

    def __init__(
        self,
        marker_ids: Sequence[int] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.marker_ids = tuple(sorted({int(marker_id) for marker_id in marker_ids})) if marker_ids is not None else None
        self.use_timestamps = bool(self.config.get("use_timestamps", True))
        self._title = str(self.config.get("title", self.DEFAULT_TITLE))

    def analyze(self, signal: ISignal) -> ArucoMarkerDisplacementData:
        samples = self._aruco_samples(signal)
        source_path, resize, fps = self._source_metadata(samples)
        marker_ids = self._resolved_marker_ids(samples)

        series: list[ArucoMarkerDisplacementSeries] = []
        for marker_id in marker_ids:
            baseline = self._baseline_center(samples, marker_id)
            if baseline is None:
                continue
            series.append(self._build_series(samples, marker_id, baseline))

        if not series:
            raise ValueError("ArucoMarkerDisplacementAnalyzer requires at least one detected marker.")

        return ArucoMarkerDisplacementData(
            series=series,
            frames=samples,
            title=self._title,
            source_path=source_path,
            resize=resize,
            fps=fps,
            metadata={
                "title": self._title,
                "marker_ids": [entry.marker_id for entry in series],
                "use_timestamps": self.use_timestamps,
            },
        )

    def _build_series(
        self,
        samples: list[ArucoMarkerSignalSample],
        marker_id: int,
        baseline: Point2D,
    ) -> ArucoMarkerDisplacementSeries:
        frame_indices: list[int] = []
        timestamps: list[float | None] = []
        detected_flags: list[bool] = []
        dx_values: list[float] = []
        dy_values: list[float] = []
        magnitude_values: list[float] = []

        for sample in samples:
            observation = sample.marker_by_id(marker_id)
            frame_indices.append(int(sample.frame_index))
            timestamps.append(sample.timestamp_seconds)
            if observation is None or not observation.detected or observation.center is None:
                detected_flags.append(False)
                dx_values.append(float("nan"))
                dy_values.append(float("nan"))
                magnitude_values.append(float("nan"))
                continue

            dx = float(observation.center[0] - baseline[0])
            dy = float(observation.center[1] - baseline[1])
            detected_flags.append(True)
            dx_values.append(dx)
            dy_values.append(dy)
            magnitude_values.append(math.hypot(dx, dy))

        return ArucoMarkerDisplacementSeries(
            marker_id=marker_id,
            frame_indices=frame_indices,
            timestamps=timestamps,
            detected=detected_flags,
            displacement_x=dx_values,
            displacement_y=dy_values,
            displacement_magnitude=magnitude_values,
            initial_center=baseline,
            stats=self._stats(dx_values, dy_values, magnitude_values, detected_flags),
            metadata={"use_timestamps": self.use_timestamps},
        )

    def _resolved_marker_ids(self, samples: list[ArucoMarkerSignalSample]) -> tuple[int, ...]:
        if self.marker_ids is not None:
            return self.marker_ids
        detected_marker_ids = sorted(
            {
                marker.marker_id
                for sample in samples
                for marker in sample.markers
                if marker.detected
            }
        )
        return tuple(detected_marker_ids)

    @staticmethod
    def _baseline_center(
        samples: list[ArucoMarkerSignalSample],
        marker_id: int,
    ) -> Point2D | None:
        for sample in samples:
            observation = sample.marker_by_id(marker_id)
            if observation is not None and observation.detected and observation.center is not None:
                return observation.center
        return None

    @staticmethod
    def _aruco_samples(signal: ISignal) -> list[ArucoMarkerSignalSample]:
        samples = list(signal)
        if any(not isinstance(sample, ArucoMarkerSignalSample) for sample in samples):
            raise TypeError("ArucoMarkerDisplacementAnalyzer requires ArucoMarkerSignalSample inputs.")
        return samples

    @staticmethod
    def _source_metadata(
        samples: list[ArucoMarkerSignalSample],
    ) -> tuple[str | None, tuple[int, int] | None, float | None]:
        source_path: str | None = None
        resize: tuple[int, int] | None = None
        fps: float | None = None
        for sample in samples:
            metadata = dict(sample.metadata)
            if source_path is None:
                raw_source_path = metadata.get("source_path")
                if raw_source_path:
                    source_path = str(raw_source_path)
            if resize is None:
                raw_resize = metadata.get("resize")
                if isinstance(raw_resize, (tuple, list)) and len(raw_resize) == 2:
                    resize = (int(raw_resize[0]), int(raw_resize[1]))
            if fps is None:
                raw_fps = metadata.get("source_fps")
                try:
                    fps_value = float(raw_fps) if raw_fps is not None else None
                except (TypeError, ValueError):
                    fps_value = None
                if fps_value is not None and fps_value > 0:
                    fps = fps_value
        return source_path, resize, fps

    @staticmethod
    def _stats(
        dx_values: list[float],
        dy_values: list[float],
        magnitude_values: list[float],
        detected_flags: list[bool],
    ) -> dict[str, float]:
        finite_dx = [value for value in dx_values if math.isfinite(value)]
        finite_dy = [value for value in dy_values if math.isfinite(value)]
        finite_magnitudes = [value for value in magnitude_values if math.isfinite(value)]
        return {
            "detected_samples": float(sum(detected_flags)),
            "max_abs_dx": max((abs(value) for value in finite_dx), default=0.0),
            "max_abs_dy": max((abs(value) for value in finite_dy), default=0.0),
            "max_magnitude": max(finite_magnitudes, default=0.0),
            "mean_magnitude": (sum(finite_magnitudes) / len(finite_magnitudes)) if finite_magnitudes else 0.0,
        }
