from abc import ABC, abstractmethod
from typing import Any, Dict

from library.core.artifacts.FrameSequenceBuffer import FrameSequenceBuffer
from library.core.artifacts.Signal import Signal

class ITracker(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def track(self, frames_buffer: FrameSequenceBuffer) -> Signal:
        pass