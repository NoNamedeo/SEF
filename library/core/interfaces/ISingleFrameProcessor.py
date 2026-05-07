from abc import ABC, abstractmethod
from typing import Any

from library.core.artifacts.Frame import Frame


class ISingleFrameProcessor(ABC):
    """Process one frame without requiring sequence-level context."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def process(self, frame: Frame) -> Frame:
        """Return the processed frame."""
