from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.artifacts.buffer.FrameBuffer import FrameBuffer
from library.core.interfaces.StageCapabilities import StageCapabilities


class IFrameBufferProcessor(ABC):
    """
    Buffer-level frame processing contract.

    Implementations receive the whole frame sequence so both stateless
    per-frame operations and temporal algorithms can participate in the same
    frame processing pipeline.

    Streaming implementations should also implement
    `IStreamingFrameBufferProcessor` when they can publish processed frames
    progressively without requiring the complete sequence first.
    """

    capabilities = StageCapabilities.batch()

    def __init__(self, config: dict[str, Any] | None = None):
        """Store plugin-specific processor configuration."""
        self.config = config or {}

    @abstractmethod
    def process(self, buffer: FrameBuffer) -> FrameBuffer:
        """
        Return a processed frame buffer.

        Parameters
        ----------
        buffer:
            Input frames in pipeline order.

        Returns
        -------
        FrameBuffer
            Processed frames. Implementations should preserve order unless
            documented otherwise.
        """
