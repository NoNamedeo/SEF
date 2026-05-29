from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from library.core.interfaces.ISignalSample import ISignalSample


Skeleton2D = np.ndarray  # shape: (17, 2)
Confidence = np.ndarray   # shape: (17,)


@dataclass(slots=True)
class COCOSkeletonSignalSample(ISignalSample):
    """
    Single COCO pose observation extracted from one frame.

    Expected format:
        skeleton: [17, 2] (x, y coordinates)
        confidence: [17] (per-joint confidence)
    """

    frame_index: int

    skeleton: Skeleton2D | None
    confidence: Confidence | None

    centroid: tuple[float, float] | None = None
    timestamp_seconds: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)