from abc import ABC, abstractmethod
from typing import Iterable, Any, Dict

class ITracker(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def track(self, frames: Iterable[Any]):

        pass