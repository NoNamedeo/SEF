from abc import ABC, abstractmethod
from typing import Any, Dict

from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.StageCapabilities import StageCapabilities


class ISignalCleaner(ABC):
    """
    Batch contract for transforming signal samples before analysis.

    Cleaners should preserve signal semantics unless their documentation states
    otherwise. Typical implementations smooth, normalize, filter, or repair
    samples produced by a signal extractor.
    """

    capabilities = StageCapabilities.batch()

    def __init__(self, config: Dict[str, Any] | None = None):
        """Store plugin-specific cleaner configuration."""
        self.config = config or {}

    @abstractmethod
    def clean(self, signal: ISignal) -> ISignal:
        """
        Return a cleaned signal.

        Parameters
        ----------
        signal:
            Input signal from the previous extraction or cleaning stage.

        Returns
        -------
        ISignal
            Cleaned signal for the next cleaner or analyzer.
        """
