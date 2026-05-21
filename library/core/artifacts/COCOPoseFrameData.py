from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from library.core.interfaces.IData import IData


@dataclass(slots=True)
class COCOPoseFrameData(IData):
    """Visualization-ready COCO pose observation for one frame."""

    frame_index: int
    skeleton: np.ndarray | None
    confidence: np.ndarray | None
    centroid: tuple[float, float] | None = None
    timestamp_seconds: float | None = None
    frame_size: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class COCOPoseSequenceData(IData):
    """Batch container returned after consuming a COCO pose stream."""

    frames: list[COCOPoseFrameData] = field(default_factory=list)
    title: str = "COCO Pose Sequence"
    metadata: dict[str, Any] = field(default_factory=dict)
