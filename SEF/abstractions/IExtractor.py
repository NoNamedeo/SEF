from abc import ABC, abstractmethod
from typing import Any, Dict

class IExtractor(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def extract(self, video):
        pass