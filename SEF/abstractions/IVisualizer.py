from abc import ABC, abstractmethod
from typing import Any, Dict

from SEF.artifacts.Data import Data

class IVisualizer(ABC):

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def visualize(self, data: Data):
        pass