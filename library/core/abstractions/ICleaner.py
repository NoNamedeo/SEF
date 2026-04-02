from abc import ABC, abstractmethod
from typing import Any, Dict

from library.core.artifacts.Signal import Signal

class ICleaner(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def clean(self, signal: Signal) -> Signal:
        pass