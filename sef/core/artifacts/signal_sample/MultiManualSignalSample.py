from __future__ import annotations

from dataclasses import dataclass, field

from sef.core.artifacts.signal_sample.BoxSignalSample import BoxSignalSample
from sef.core.interfaces.ISignalSample import ISignalSample

BoundingBox = tuple[int, int, int, int]
Point2D = tuple[float, float]


@dataclass(slots=True)
class MultiManualSignalSample(ISignalSample):
    """multiple signal observation extracted from one frame."""
    samples: dict[int, BoxSignalSample] = field(default_factory=dict)
