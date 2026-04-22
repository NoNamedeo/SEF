from __future__ import annotations

from dataclasses import dataclass, field
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
