from __future__ import annotations

from sef.core.artifacts.Signal import Signal
from sef.core.artifacts.signal_sample.BoxSignalSample import BoxSignalSample
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalCleaner import ISignalCleaner


class MovingAverageCleaner(ISignalCleaner):
    """Smooth centroid coordinates with a moving average window."""

    def __init__(self, window_size: int, config=None):
        super().__init__(config)
        self.window_size = int(window_size)

        if self.window_size <= 0:
            raise ValueError("window_size must be greater than 0")

    def clean(self, signal: ISignal) -> ISignal:
        samples = list(signal)
        cleaned_samples: list[BoxSignalSample] = []
        centroids = [sample.centroid for sample in samples]

        for index, sample in enumerate(samples):
            start = max(0, index - self.window_size // 2)
            end = min(len(samples), index + self.window_size // 2 + 1)

            window_points = [point for point in centroids[start:end] if point is not None]
            smoothed_centroid = None

            if window_points:
                avg_x = sum(point[0] for point in window_points) / len(window_points)
                avg_y = sum(point[1] for point in window_points) / len(window_points)
                smoothed_centroid = (avg_x, avg_y)

            cleaned_samples.append(
                BoxSignalSample(
                    frame_index=sample.frame_index,
                    box=sample.box,
                    centroid=smoothed_centroid,
                    timestamp_seconds=sample.timestamp_seconds,
                    metadata=dict(sample.metadata),
                )
            )

        return Signal(cleaned_samples, config=dict(signal.config))
