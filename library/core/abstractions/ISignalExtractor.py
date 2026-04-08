from abc import ABC, abstractmethod
from typing import Any, Dict

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.abstractions.ISignal import ISignal

class ISignalExtractor(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def extract(self, buffer: FrameBuffer) -> ISignal:
        pass
