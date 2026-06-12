from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.interfaces.IData import IData
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.StageCapabilities import StageCapabilities


class IAnalyzer(ABC):
    """
    Batch contract for turning a signal into analytical data.

    An analyzer is the boundary between signal processing and visualization.
    Streaming analyzers should also implement `IStreamingAnalyzer` to publish
    progressive `IData` values while still returning a final result.
    """

    capabilities = StageCapabilities.batch()

    def __init__(self, config: dict[str, Any] | None = None):
        """Store plugin-specific analyzer configuration."""
        self.config = config or {}

    @abstractmethod
    def analyze(self, signal: ISignal) -> IData:
        """
        Analyze a complete signal.

        Parameters
        ----------
        signal:
            Signal after all configured cleaners have run.

        Returns
        -------
        IData
            UI-agnostic analytical result.
        """
