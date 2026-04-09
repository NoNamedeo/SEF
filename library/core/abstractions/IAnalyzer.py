from abc import ABC, abstractmethod
from typing import Any, Dict

from library.core.abstractions.ISignal import ISignal
from library.core.artifacts.Data import Data

class IAnalyzer(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def analyze(self, signal: ISignal):
        pass