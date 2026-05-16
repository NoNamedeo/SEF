from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.interfaces.ISignal import ISignal
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.StageCapabilities import StageCapabilities


class ISignalExtractor(ABC):
    capabilities = StageCapabilities.batch()

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def extract(self, buffer: FrameBuffer) -> ISignal:
        """Extract a signal from the provided frame buffer."""
