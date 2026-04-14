from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from library.core.interfaces.ISignalSample import ISignalSample

BoundingBox = tuple[int, int, int, int]
Point2D = tuple[float, float]


@dataclass(slots=True)
class BoxSignalSample(ISignalSample):
    """Single signal observation extracted from one frame."""

    frame_index: int
    box: BoundingBox | None
    centroid: Point2D | None
    timestamp_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
