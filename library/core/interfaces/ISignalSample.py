from __future__ import annotations

from abc import ABC
from typing import Any

class ISignalSample(ABC):
    """
    interface for signal samples.
    """

    def __init__(
        self,
        frame_index: int,
        timestamp_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.frame_index = frame_index
        self.timestamp_seconds = timestamp_seconds
        self.metadata = metadata or {}