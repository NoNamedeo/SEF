from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sef.core.interfaces.ISignalSample import ISignalSample


class ILiveAnalyzer(ABC):
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def update(self, signal: ISignalSample):
        """Turn an extracted signal into analytical data."""

    @abstractmethod
    def start(self):
        """Start the live plot."""
