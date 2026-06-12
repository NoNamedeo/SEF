from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from library.core.artifacts.signal_sample.BoxSignalSample import BoundingBox, Point2D
from library.core.interfaces.IData import IData


@dataclass(slots=True)
class TrackingPlaybackTrack:
    """Serializable tracking observation used to rebuild annotated video frames."""

    track_id: int
    box: BoundingBox
    centroid: Point2D | None = None
    confidence: float | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrackingPlaybackFrame:
    """Tracked objects associated with a single frame index."""

    frame_index: int
    tracks: list[TrackingPlaybackTrack]
    timestamp_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrackingPlaybackData(IData):
    """Playback-ready tracking data consumed by video visualizers."""

    frames: list[TrackingPlaybackFrame]
    title: str = "Tracking Playback"
    source_path: str | None = None
    resize: tuple[int, int] | None = None
    fps: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frames = list(self.frames)
        if self.resize is not None:
            self.resize = (int(self.resize[0]), int(self.resize[1]))

        base_metadata = dict(self.metadata)
        if self.source_path is not None:
            base_metadata.setdefault("source_path", self.source_path)
        if self.resize is not None:
            base_metadata.setdefault("resize", self.resize)
        if self.fps is not None:
            base_metadata.setdefault("fps", self.fps)
        base_metadata.setdefault("frame_count", len(self.frames))
        base_metadata.setdefault(
            "track_count",
            len(
                {
                    track.track_id
                    for frame in self.frames
                    for track in frame.tracks
                }
            ),
        )
        self.metadata = base_metadata
