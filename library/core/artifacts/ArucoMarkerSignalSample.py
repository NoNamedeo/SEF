from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from library.core.interfaces.ISignalSample import ISignalSample

Point2D = tuple[float, float]
MarkerCorners = tuple[Point2D, Point2D, Point2D, Point2D]


@dataclass(slots=True)
class ArucoMarkerObservation:
    """A single ArUco marker observation inside one frame."""

    marker_id: int
    corners: MarkerCorners | None
    center_x: float | None = None
    center_y: float | None = None
    detected: bool = True
    quality_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Point2D | None:
        if self.center_x is None or self.center_y is None:
            return None
        return self.center_x, self.center_y


@dataclass(slots=True)
class ArucoMarkerSignalSample(ISignalSample):
    """Frame-level collection of ArUco marker observations."""

    frame_index: int
    markers: list[ArucoMarkerObservation]
    timestamp_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.markers = list(self.markers)

    def marker_by_id(self, marker_id: int) -> ArucoMarkerObservation | None:
        for marker in self.markers:
            if marker.marker_id == marker_id:
                return marker
        return None
