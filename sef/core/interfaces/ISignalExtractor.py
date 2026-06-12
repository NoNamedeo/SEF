from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.StageCapabilities import StageCapabilities


class ISignalExtractor(ABC):
    """
    Batch contract for converting frames into signal samples.

    Signal extractors consume processed frames and produce an `ISignal` that
    analyzers can consume. Streaming implementations should also implement
    `IStreamingSignalExtractor` and declare streaming capabilities.
    """

    capabilities = StageCapabilities.batch()

    def __init__(self, config: dict[str, Any] | None = None):
        """Store plugin-specific extractor configuration."""
        self.config = config or {}

    @abstractmethod
    def extract(self, buffer: FrameBuffer) -> ISignal:
        """
        Extract a signal from a frame buffer.

        Parameters
        ----------
        buffer:
            Processed frames in pipeline order.

        Returns
        -------
        ISignal
            Iterable signal samples preserving frame-time semantics when
            available.
        """
