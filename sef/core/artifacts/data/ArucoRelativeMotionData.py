from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sef.core.interfaces.IData import IData

MarkerPair = tuple[int, int]


@dataclass(slots=True)
class ArucoMarkerRelativeMotionSeries:
    """Distance variation timeline between two markers."""

    marker_pair: MarkerPair
    frame_indices: list[int]
    timestamps: list[float | None]
    detected: list[bool]
    distances: list[float]
    distance_deltas: list[float]
    baseline_distance: float
    stats: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArucoMarkerRelativeMotionData(IData):
    """Relative motion analysis across one or more marker pairs."""

    series: list[ArucoMarkerRelativeMotionSeries]
    title: str = "ArUco Relative Motion"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.series = list(self.series)
        base_metadata = dict(self.metadata)
        base_metadata.setdefault("pair_count", len(self.series))
        self.metadata = base_metadata
