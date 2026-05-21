from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from library.core.artifacts.BoxSignalSample import BoxSignalSample
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.BufferContracts import IBuffer
from library.core.interfaces.ISignalSample import ISignalSample
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import IStreamingSignalCleaner


class MovingAverageStreamCleaner(IStreamingSignalCleaner):
    """Smooth centroid coordinates with a moving average window over a SignalBuffer."""

    capabilities = StageCapabilities.streaming(
        stateful=True,
        preserves_order=True,
        realtime_safe=True,
    )

    def __init__(self, window_size: int, buffer: SignalBuffer = None, config=None):
        super().__init__(config)

        self.window_size = int(window_size)

        if self.window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        self._default_buffer = buffer

    def clean(self, input_buffer: SignalBuffer) -> SignalBuffer:
        """
        Smooth centroids with a causal moving average and stream results.

        A centered moving average requires future samples and therefore forces a
        full materialization. For realtime streaming this cleaner intentionally
        uses the last ``window_size`` available samples.
        """
        output = self._default_buffer or SignalBuffer()
        self.clean_into(input_buffer, output)
        return output

    def clean_into(self, input_buffer: Iterable[ISignalSample], output_buffer: IBuffer[ISignalSample]) -> None:
        window = deque(maxlen=self.window_size)

        try:
            for sample in input_buffer:
                if not isinstance(sample, BoxSignalSample):
                    raise TypeError("MovingAverageStreamCleaner requires BoxSignalSample inputs.")
                if sample.centroid is not None:
                    window.append(sample.centroid)

                smoothed_centroid = None
                if window:
                    avg_x = sum(point[0] for point in window) / len(window)
                    avg_y = sum(point[1] for point in window) / len(window)

                    smoothed_centroid = (avg_x, avg_y)

                cleaned_sample = BoxSignalSample(
                    frame_index=sample.frame_index,
                    box=sample.box,
                    centroid=smoothed_centroid,
                    timestamp_seconds=sample.timestamp_seconds,
                    metadata=dict(sample.metadata),
                )

                output_buffer.put(cleaned_sample)
        finally:
            output_buffer.close()
