from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from library.core.abstractions.ISignalSample import ISignalSample

BoundingBox = tuple[int, int, int, int]
Point2D = tuple[float, float]
Vector2D = tuple[float, float]


@dataclass(slots=True)
class SparseOpticalFlowSignalSample(ISignalSample):
    """Enhanced signal sample for optical-flow-based extraction."""

    frame_index: int
    box: BoundingBox | None
    points: list[Point2D] = field(default_factory=list)
    point_vectors: list[Vector2D] = field(default_factory=list)
    motion_vector: Vector2D | None = None
    motion_magnitude: float | None = None
    motion_angle: float | None = None
    timestamp_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
