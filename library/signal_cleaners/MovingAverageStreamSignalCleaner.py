from __future__ import annotations

from collections import deque

from library.core.artifacts.BoxSignalSample import BoxSignalSample
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.ISignalCleaner import ISignalCleaner


class MovingAverageStreamCleaner(ISignalCleaner):
    """Smooth centroid coordinates with a moving average window over a SignalBuffer."""

    def __init__(self, window_size: int, buffer: SignalBuffer = None, config=None):
        super().__init__(config)

        self.window_size = int(window_size)

        if self.window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        # Output buffer
        self.buffer = buffer or SignalBuffer()

    def clean(self, input_buffer: SignalBuffer) -> SignalBuffer:
        """
        Smooth centroids with a causal moving average and stream results.

        A centered moving average requires future samples and therefore forces a
        full materialization. For realtime streaming this cleaner intentionally
        uses the last ``window_size`` available samples.
        """
        window = deque(maxlen=self.window_size)

        for sample in input_buffer:
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

            self.buffer.put(cleaned_sample)

        self.buffer.close()
        return self.buffer
