from abc import ABC, abstractmethod
from typing import Any, Dict

class ICleaner(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def clean(self, data):
        pass