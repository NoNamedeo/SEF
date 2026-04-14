from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.interfaces.IData import IData


class IVisualizer(ABC):
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def visualize(self, data: IData):
        """Render analytical data."""
