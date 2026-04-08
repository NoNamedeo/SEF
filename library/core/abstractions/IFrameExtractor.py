from abc import ABC, abstractmethod
from typing import Any, Dict

from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.artifacts.CompositeFrameCleaner import CompositeFrameCleaner


class IFrameExtractor(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def extract(self, composite_frame_cleaner: CompositeFrameCleaner):
        pass