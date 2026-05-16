from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.IData import IData
from library.core.interfaces.StageCapabilities import StageCapabilities


class IAnalyzer(ABC):
    capabilities = StageCapabilities.batch()

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def analyze(self, signal: ISignal) -> IData:
        """Turn an extracted signal into analytical data."""
