from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.StageCapabilities import StageCapabilities


class IFrameExtractor(ABC):
    capabilities = StageCapabilities.batch()

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def extract(self) -> FrameBuffer:
        """Extract raw frames and return them in a buffer."""
