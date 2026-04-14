from abc import ABC, abstractmethod
from typing import Any, Dict

from library.core.artifacts.Frame import Frame


class IFrameCleaner(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def clean(self, frame: Frame) -> Frame:
        pass