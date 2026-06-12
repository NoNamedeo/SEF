from __future__ import annotations

from abc import ABC
from typing import Any


class ISignalSample(ABC):
    """
    Base contract for one time-indexed signal sample.

    Attributes
    ----------
    frame_index:
        Source frame index associated with the sample.
    timestamp_seconds:
        Optional source timestamp in seconds.
    metadata:
        Plugin-specific sample metadata.
    """

    def __init__(
        self,
        frame_index: int,
        timestamp_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Create a signal sample value."""
        self.frame_index = frame_index
        self.timestamp_seconds = timestamp_seconds
        self.metadata = metadata or {}
