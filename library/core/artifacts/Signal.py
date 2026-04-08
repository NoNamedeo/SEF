from library.core.abstractions.ISignal import ISignal
from typing import Any, Dict

class Signal(ISignal):

    def __init__(self, signal, config: Dict[str, Any] | None = None):
        super().__init__(config, signal)