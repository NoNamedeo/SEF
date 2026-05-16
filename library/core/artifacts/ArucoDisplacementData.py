from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import math
from typing import Any

from library.core.artifacts.ArucoMarkerSignalSample import ArucoMarkerSignalSample, Point2D
from library.core.interfaces.IData import IData


@dataclass(slots=True)
class ArucoMarkerDisplacementSeries:
    """Displacement timeline for a single marker."""

    marker_id: int
    frame_indices: list[int]
    timestamps: list[float | None]
    detected: list[bool]
    displacement_x: list[float]
    displacement_y: list[float]
    displacement_magnitude: list[float]
    initial_center: Point2D
    stats: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArucoMarkerDisplacementObservation:
    """Progressive displacement value for one marker in one frame."""

    marker_id: int
    detected: bool
    displacement_x: float
    displacement_y: float
    displacement_magnitude: float
    initial_center: Point2D | None


@dataclass(slots=True)
class ArucoMarkerDisplacementFrameData(IData):
    """Streaming displacement payload emitted once per ArUco signal sample."""

    frame: ArucoMarkerSignalSample
    displacements: dict[int, ArucoMarkerDisplacementObservation]
    title: str = "ArUco Marker Displacement"
    source_path: str | None = None
    resize: tuple[int, int] | None = None
    fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.displacements = dict(self.displacements)
        if self.resize is not None:
            self.resize = (int(self.resize[0]), int(self.resize[1]))
        base_metadata = dict(self.metadata)
        base_metadata.setdefault("frame_index", int(self.frame.frame_index))
        base_metadata.setdefault("marker_count", len(self.displacements))
        if self.source_path is not None:
            base_metadata.setdefault("source_path", self.source_path)
        if self.resize is not None:
            base_metadata.setdefault("resize", self.resize)
        if self.fps is not None:
            base_metadata.setdefault("fps", self.fps)
        self.metadata = base_metadata


@dataclass(slots=True)
class ArucoMarkerDisplacementData(IData):
    """Playback-ready displacement analysis for multiple ArUco markers."""

    series: list[ArucoMarkerDisplacementSeries]
    frames: list[ArucoMarkerSignalSample]
    title: str = "ArUco Marker Displacement"
    source_path: str | None = None
    resize: tuple[int, int] | None = None
    fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.series = list(self.series)
        self.frames = list(self.frames)
        if self.resize is not None:
            self.resize = (int(self.resize[0]), int(self.resize[1]))

        base_metadata = dict(self.metadata)
        base_metadata.setdefault("frame_count", len(self.frames))
        base_metadata.setdefault("marker_count", len(self.series))
        if self.source_path is not None:
            base_metadata.setdefault("source_path", self.source_path)
        if self.resize is not None:
            base_metadata.setdefault("resize", self.resize)
        if self.fps is not None:
            base_metadata.setdefault("fps", self.fps)
        self.metadata = base_metadata

    @classmethod
    def from_progressive_frames(
        cls,
        frames: list[ArucoMarkerDisplacementFrameData],
        *,
        title: str = "ArUco Marker Displacement",
        use_timestamps: bool = True,
    ) -> ArucoMarkerDisplacementData:
        """Build the final displacement result from progressive frame payloads."""
        marker_ids = sorted({marker_id for frame in frames for marker_id in frame.displacements})
        series: list[ArucoMarkerDisplacementSeries] = []
        for marker_id in marker_ids:
            frame_indices: list[int] = []
            timestamps: list[float | None] = []
            detected_flags: list[bool] = []
            dx_values: list[float] = []
            dy_values: list[float] = []
            magnitude_values: list[float] = []
            initial_center: Point2D | None = None

            for frame_data in frames:
                observation = frame_data.displacements.get(marker_id)
                frame_indices.append(int(frame_data.frame.frame_index))
                timestamps.append(frame_data.frame.timestamp_seconds)
                if observation is None:
                    detected_flags.append(False)
                    dx_values.append(float("nan"))
                    dy_values.append(float("nan"))
                    magnitude_values.append(float("nan"))
                    continue
                if initial_center is None and observation.initial_center is not None:
                    initial_center = observation.initial_center
                detected_flags.append(observation.detected)
                dx_values.append(float(observation.displacement_x))
                dy_values.append(float(observation.displacement_y))
                magnitude_values.append(float(observation.displacement_magnitude))

            if initial_center is None:
                continue
            series.append(
                ArucoMarkerDisplacementSeries(
                    marker_id=marker_id,
                    frame_indices=frame_indices,
                    timestamps=timestamps,
                    detected=detected_flags,
                    displacement_x=dx_values,
                    displacement_y=dy_values,
                    displacement_magnitude=magnitude_values,
                    initial_center=initial_center,
                    stats=_displacement_stats(dx_values, dy_values, magnitude_values, detected_flags),
                    metadata={"use_timestamps": use_timestamps},
                )
            )

        source_path = next((frame.source_path for frame in frames if frame.source_path), None)
        resize = next((frame.resize for frame in frames if frame.resize is not None), None)
        fps = next((frame.fps for frame in frames if frame.fps is not None), None)
        return cls(
            series=series,
            frames=[frame.frame for frame in frames],
            title=title,
            source_path=source_path,
            resize=resize,
            fps=fps,
            metadata={
                "title": title,
                "marker_ids": [entry.marker_id for entry in series],
                "use_timestamps": use_timestamps,
                "progressive_frames": len(frames),
            },
        )

    @classmethod
    def from_stream_items(
        cls,
        data: Iterable[IData],
        *,
        title: str | None = None,
        use_timestamps: bool | None = None,
    ) -> ArucoMarkerDisplacementData:
        """
        Normalize progressive analyzer output into the final displacement model.

        Streaming visualizers receive per-frame payloads. Batch visualizers receive
        the final aggregate directly. Keeping this conversion here prevents each
        visualizer from duplicating stream reconstruction rules.
        """
        progressive_frames: list[ArucoMarkerDisplacementFrameData] = []
        final_data: ArucoMarkerDisplacementData | None = None

        for item in data:
            if isinstance(item, ArucoMarkerDisplacementData):
                final_data = item
                continue
            if isinstance(item, ArucoMarkerDisplacementFrameData):
                progressive_frames.append(item)
                continue
            raise TypeError(
                "ArUco displacement stream requires ArucoMarkerDisplacementFrameData "
                f"or ArucoMarkerDisplacementData items, got {type(item).__name__}."
            )

        if final_data is not None:
            return final_data
        if not progressive_frames:
            return cls(
                series=[],
                frames=[],
                title=title or "ArUco Marker Displacement",
                metadata={"progressive_frames": 0},
            )

        first_metadata = progressive_frames[0].metadata
        resolved_title = title or str(first_metadata.get("title", "ArUco Marker Displacement"))
        resolved_use_timestamps = (
            bool(first_metadata.get("use_timestamps", True))
            if use_timestamps is None
            else bool(use_timestamps)
        )
        return cls.from_progressive_frames(
            progressive_frames,
            title=resolved_title,
            use_timestamps=resolved_use_timestamps,
        )


def _displacement_stats(
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
