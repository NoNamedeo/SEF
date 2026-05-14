from __future__ import annotations

from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.NoData import NoData
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.ISignal import ISignal


class NoAnalyzer(IAnalyzer):
    """No operation analyzer."""

    def __init__(self, config=None):
        super().__init__(config)
        self.buffer = DataBuffer(buffer_size=1)

    def analyze(self, signal: ISignal) -> IData:
        for _sample in signal:
            pass
        self.buffer.close()
        return NoData()
