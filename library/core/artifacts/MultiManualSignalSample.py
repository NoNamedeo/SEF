from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from library.core.interfaces.ISignalSample import ISignalSample
from library.core.artifacts.BoxSignalSample import BoxSignalSample

BoundingBox = tuple[int, int, int, int]
Point2D = tuple[float, float]


@dataclass(slots=True)
class MultiManualSignalSample(ISignalSample):
    """multiple signal observation extracted from one frame."""
    samples: dict[int, BoxSignalSample] = field(default_factory=dict)
