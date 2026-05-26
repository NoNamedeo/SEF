from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.StageCapabilities import StageCapabilities


class IFrameExtractor(ABC):
    """
    Batch source contract for frame-producing plugins.

    A frame extractor is the pipeline entry point. Batch extractors return a
    closed `FrameBuffer`; streaming extractors should additionally implement
    `IStreamingFrameExtractor` so the planner can avoid materializing source
    frames.

    Extension guidance
    ------------------
    Implementations should keep constructor configuration explicit and should
    encode frame order through `Frame.index` when available.
    """

    capabilities = StageCapabilities.batch()

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Store optional extractor configuration.

        Parameters
        ----------
        config:
            Plugin-specific configuration mapping. The core does not interpret
            this mapping after construction.
        """
        self.config = config or {}

    @abstractmethod
    def extract(self) -> FrameBuffer:
        """
        Extract raw frames and return them in a closed buffer.

        Returns
        -------
        FrameBuffer
            Buffer containing frames in source order.
        """
