from __future__ import annotations

from typing import Any

from sef.core.artifacts.signal_sample.BoxSignalSample import BoxSignalSample
from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.artifacts.Signal import Signal
from sef.core.artifacts.buffer.SignalBuffer import SignalBuffer
from sef.core.interfaces.BufferContracts import IBuffer, IFrameBuffer
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalSample import ISignalSample
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.interfaces.StreamingContracts import IStreamingSignalExtractor


class NoSignalExtractor(IStreamingSignalExtractor):
    """No operation tracker."""

    capabilities = StageCapabilities.streaming(
        stateful=False,
        preserves_order=True,
        realtime_safe=True,
    )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples = SignalBuffer(buffer_size=1)
        self.extract_into(buffer, samples)
        return Signal(list(samples))

    def extract_into(self, frames: IFrameBuffer, output_buffer: IBuffer[ISignalSample]) -> None:
        last_frame_index = 0
        last_timestamp = None
        for frame in frames:
            last_frame_index = frame.index or 0
            last_timestamp = frame.timestamp_seconds
        sample = BoxSignalSample(
            frame_index=last_frame_index,
            box=(0, 0, 0, 0),
            centroid=(0, 0),
            timestamp_seconds=last_timestamp,
        )
        output_buffer.put(sample)
        output_buffer.close()
