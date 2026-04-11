from __future__ import annotations

import numpy as np

from library.core.abstractions.ISignal import ISignal
from library.core.abstractions.ISignalCleaner import ISignalCleaner
from library.core.artifacts.Signal import Signal
from library.core.artifacts.BoxSignalSample import BoxSignalSample


class OutlierRejectionCleaner(ISignalCleaner):
    """Remove or replace outliers in centroid signal using MAD (robust)."""

    def __init__(self, threshold: float = 3.5, mode: str = "clip", config=None):
        """
        mode:
            - "clip" → porta outlier al limite
            - "remove" → mette centroid=None
            - "replace" → sostituisce con mediana
        """
        super().__init__(config)
        self.threshold = float(threshold)
        self.mode = mode

    def clean(self, signal: ISignal) -> ISignal:
        samples = list(signal)
        cleaned_samples: list[BoxSignalSample] = []

        centroids = np.array([
            np.asarray(s.centroid, dtype=float)
            for s in samples
            if s.centroid is not None
        ])

        if len(centroids) == 0:
            return signal

        median = np.median(centroids, axis=0)

        diff = np.abs(centroids - median)
        mad = np.median(diff, axis=0)

        mad[mad == 0] = 1e-6

        for sample in samples:
            if sample.centroid is None:
                cleaned_samples.append(sample)
                continue

            c = np.asarray(sample.centroid, dtype=float)

            z = np.abs(c - median) / mad

            is_outlier = np.any(z > self.threshold)

            if not is_outlier:
                new_centroid = c
            else:
                if self.mode == "remove":
                    new_centroid = None

                elif self.mode == "replace":
                    new_centroid = median

                elif self.mode == "clip":
                    clipped = np.clip(c, median - self.threshold * mad,
                                        median + self.threshold * mad)
                    new_centroid = clipped

                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

            cleaned_samples.append(
                BoxSignalSample(
                    frame_index=sample.frame_index,
                    box=sample.box,
                    centroid=new_centroid,
                    timestamp_seconds=sample.timestamp_seconds,
                    metadata=dict(sample.metadata),
                )
            )

        return Signal(cleaned_samples, config=dict(signal.config))