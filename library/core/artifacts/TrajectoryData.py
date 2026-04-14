from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from library.core.abstractions.IData import IData


@dataclass(slots=True)
class TrajectoryData(IData):
    """
    Represents trajectories of multiple tracked points over time.

    Each trajectory is a sequence of (x, y) positions.
    """

    trajectories_x: list[list[float]]  # [track_id][time]
    trajectories_y: list[list[float]]  # [track_id][time]

    frame_indices: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)