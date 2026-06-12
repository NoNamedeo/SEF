from __future__ import annotations

from collections.abc import Iterable

from sef.core.artifacts.buffer.DataBuffer import DataBuffer
from sef.core.artifacts.data.NoData import NoData
from sef.core.interfaces.BufferContracts import IBuffer
from sef.core.interfaces.IData import IData
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalSample import ISignalSample
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.interfaces.StreamingContracts import IStreamingAnalyzer


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
