from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.artifacts.FrameBuffer import FrameBuffer


class IFrameBufferProcessor(ABC):
    """
    Buffer-level frame processing contract.

    Implementations receive the whole frame sequence so both stateless
    per-frame operations and temporal algorithms can participate in the same
    frame processing pipeline.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def process(self, buffer: FrameBuffer) -> FrameBuffer:
        """Return a processed frame buffer preserving frame ordering."""
