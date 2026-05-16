from abc import ABC, abstractmethod
from typing import Any, Dict

from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.StageCapabilities import StageCapabilities


class ISignalCleaner(ABC):
    capabilities = StageCapabilities.batch()

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def clean(self, signal: ISignal) -> ISignal:
        pass
