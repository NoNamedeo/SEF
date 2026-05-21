from __future__ import annotations

from collections.abc import Iterable

from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.NoData import NoData
from library.core.interfaces.BufferContracts import IBuffer
from library.core.interfaces.IData import IData
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalSample import ISignalSample
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import IStreamingAnalyzer


class NoAnalyzer(IStreamingAnalyzer):
    """No operation analyzer."""

    capabilities = StageCapabilities.streaming(
        stateful=False,
        preserves_order=True,
        realtime_safe=True,
    )

    def __init__(self, config=None):
        super().__init__(config)

    def analyze(self, signal: ISignal) -> IData:
        return self.analyze_into(signal, DataBuffer(buffer_size=1))

    def analyze_into(self, signal: Iterable[ISignalSample], output_buffer: IBuffer[IData]) -> IData:
        for _sample in signal:
            pass
        output_buffer.close()
        return NoData()
