from abc import ABC, abstractmethod
from typing import Any, Dict

class ISignal(ABC):
    def __init__(self, signal, config: Dict[str, Any] | None = None):
        self.signal = signal
        self.config = config