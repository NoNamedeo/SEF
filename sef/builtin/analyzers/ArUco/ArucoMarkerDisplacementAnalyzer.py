from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from sef.core.artifacts.buffer.DataBuffer import DataBuffer
from sef.core.artifacts.data.ArucoDisplacementData import (
    ArucoMarkerDisplacementData,
    ArucoMarkerDisplacementFrameData,
    ArucoMarkerDisplacementObservation,
)
from sef.core.artifacts.signal_sample.ArucoMarkerSignalSample import ArucoMarkerSignalSample, Point2D
from sef.core.interfaces.BufferContracts import IBuffer
from sef.core.interfaces.IData import IData
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalSample import ISignalSample
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.interfaces.StreamingContracts import IStreamingAnalyzer


class ArucoMarkerDisplacementAnalyzer(IStreamingAnalyzer):
    """Compute 2D displacement over time for each detected marker."""

    capabilities = StageCapabilities.streaming(
        stateful=True,
        preserves_order=True,
        realtime_safe=True,
    )

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
        return self.analyze_into(signal, DataBuffer(consumers=0))

    def analyze_into(
        self,
        signal: Iterable[ISignalSample],
        output_buffer: IBuffer[IData],
    ) -> ArucoMarkerDisplacementData:
        progressive_frames: list[ArucoMarkerDisplacementFrameData] = []
        baselines: dict[int, Point2D] = {}
        known_marker_ids = set(self.marker_ids or ())

        try:
            for raw_sample in signal:
                if not isinstance(raw_sample, ArucoMarkerSignalSample):
                    raise TypeError("ArucoMarkerDisplacementAnalyzer requires ArucoMarkerSignalSample inputs.")

                known_marker_ids.update(
                    marker.marker_id for marker in raw_sample.markers if marker.detected
                )
                frame_data = self._build_progressive_frame(
                    raw_sample,
                    marker_ids=tuple(sorted(known_marker_ids)),
                    baselines=baselines,
                )
                progressive_frames.append(frame_data)
                output_buffer.put(frame_data)
        finally:
            output_buffer.close()

        result = ArucoMarkerDisplacementData.from_progressive_frames(
            progressive_frames,
            title=self._title,
            use_timestamps=self.use_timestamps,
        )
        if not result.series:
            raise ValueError("ArucoMarkerDisplacementAnalyzer requires at least one detected marker.")
        return result

    def _build_progressive_frame(
        self,
        sample: ArucoMarkerSignalSample,
        *,
        marker_ids: tuple[int, ...],
        baselines: dict[int, Point2D],
    ) -> ArucoMarkerDisplacementFrameData:
        displacements: dict[int, ArucoMarkerDisplacementObservation] = {}

        for marker_id in marker_ids:
            observation = sample.marker_by_id(marker_id)
            if observation is not None and observation.detected and observation.center is not None:
                baselines.setdefault(marker_id, observation.center)

            baseline = baselines.get(marker_id)
            if baseline is None or observation is None or not observation.detected or observation.center is None:
                displacements[marker_id] = ArucoMarkerDisplacementObservation(
                    marker_id=marker_id,
                    detected=False,
                    displacement_x=float("nan"),
                    displacement_y=float("nan"),
                    displacement_magnitude=float("nan"),
                    initial_center=baseline,
                )
                continue

            dx = float(observation.center[0] - baseline[0])
            dy = float(observation.center[1] - baseline[1])
            displacements[marker_id] = ArucoMarkerDisplacementObservation(
                marker_id=marker_id,
                detected=True,
                displacement_x=dx,
                displacement_y=dy,
                displacement_magnitude=math.hypot(dx, dy),
                initial_center=baseline,
            )

        source_path, resize, fps = self._source_metadata([sample])
        return ArucoMarkerDisplacementFrameData(
            frame=sample,
            displacements=displacements,
            title=self._title,
            source_path=source_path,
            resize=resize,
            fps=fps,
            metadata={
                "title": self._title,
                "use_timestamps": self.use_timestamps,
                "marker_ids": list(marker_ids),
            },
        )

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
