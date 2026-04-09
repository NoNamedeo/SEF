from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.abstractions.ISignal import ISignal
from library.core.artifacts.Data import Data


class IAnalyzer(ABC):
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def analyze(self, signal: ISignal) -> Data:
        """Turn an extracted signal into analytical data."""
