from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sef.core.interfaces.ISignalSample import ISignalSample

BoundingBox = tuple[int, int, int, int]
Point2D = tuple[float, float]
Vector2D = tuple[float, float]


@dataclass(slots=True)
class DenseOpticalFlowSignalSample(ISignalSample):
    """
    Represents a dense optical flow frame aggregated into a grid.
    Each cell contains a motion vector (dx, dy).
    """

    frame_index: int

    # grid info
    grid_shape: tuple[int, int]  # (rows, cols)
    cell_size: int

    # dense field (flattened grid)
    motion_field: list[Vector2D]  # one vector per cell

    # aggregated global motion (optional convenience)
    motion_vector: Vector2D | None = None
    motion_magnitude: float | None = None
    motion_angle: float | None = None

    timestamp_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
