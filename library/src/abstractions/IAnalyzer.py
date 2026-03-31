from abc import ABC, abstractmethod
from typing import Any, Dict

class IAnalyzer(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def analyze(self, signal):
        pass