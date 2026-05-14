from __future__ import annotations

from typing import Any

from library.core.artifacts.BoxSignalSample import BoxSignalSample
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor


class NoSignalExtractor(ISignalExtractor):
    """No operation tracker."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.buffer = SignalBuffer(buffer_size=1)

    def extract(self, buffer: FrameBuffer) -> ISignal:
        last_frame_index = 0
        last_timestamp = None
        for frame in buffer:
            last_frame_index = frame.index or 0
            last_timestamp = frame.timestamp_seconds
            pass
        sample = BoxSignalSample(
            frame_index=last_frame_index,
            box=(0, 0, 0, 0),
            centroid=(0, 0),
            timestamp_seconds=last_timestamp,
        )
        self.buffer.put(sample)
        self.buffer.close()
        return Signal([sample])
