from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from library.core.interfaces.ISignalSample import ISignalSample

BoundingBox = tuple[int, int, int, int]
Point2D = tuple[float, float]
Vector2D = tuple[float, float]


@dataclass(slots=True)
class MultiObjectTrack:
    """Represents a single tracked object inside a frame."""

    track_id: int
    box: BoundingBox | None
    centroid: Point2D | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MultiObjectSignalSample(ISignalSample):
    """
    A signal sample that contains multiple tracked objects per frame.
    """

    frame_index: int
    tracks: list[MultiObjectTrack]

    timestamp_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
