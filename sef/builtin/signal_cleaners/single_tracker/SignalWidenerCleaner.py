from __future__ import annotations

import numpy as np

from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.artifacts.Signal import Signal
from library.core.artifacts.signal_sample.BoxSignalSample import BoxSignalSample


class SignalWidenerCleaner(ISignalCleaner):
    """Makes the signal movements larger"""

    def __init__(self, amplification: float, config=None):
        super().__init__(config)
        self.amplification = amplification

    def clean(self, signal: ISignal) -> ISignal:
        samples = list(signal)
        cleaned_samples: list[BoxSignalSample] = []
        centroids = np.array([np.asarray(sample.centroid, dtype=float) for sample in samples if sample.centroid is not None])
        centroids_mean = np.mean(centroids, axis=0)

        for index, sample in enumerate(samples):
            if sample.centroid is None:
                continue

            c = np.asarray(sample.centroid, dtype=float)

            delta = c - centroids_mean
            smoothed_centroid = centroids_mean + self.amplification * delta

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
