from __future__ import annotations

from typing import Any
import math

import numpy as np

from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.artifacts.Signal import Signal
from library.core.artifacts.signal_sample.SparseOpticalFlowSignalSample import SparseOpticalFlowSignalSample


class OpticalFlowOutlierFilter(ISignalCleaner):
    def __init__(self, threshold: float = 3.0, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.threshold = threshold
        self.prev_vectors: list[tuple[float, float]] | None = None

    def clean(self, signal: ISignal) -> ISignal:
        samples = list(signal)
        output: list[SparseOpticalFlowSignalSample] = []

        for sample in samples:
            vectors = sample.point_vectors

            if not vectors:
                self.prev_vectors = None
                output.append(sample)
                continue

            filtered_vectors = []

            # se ho stato precedente: confronto per punto
            # TODO: e se cambio video? ho sotto reset, ma c'è un modo più furbo?
            if self.prev_vectors is not None and len(self.prev_vectors) == len(vectors):
                for (dx, dy), (pdx, pdy) in zip(vectors, self.prev_vectors):
                    dist = math.sqrt((dx - pdx) ** 2 + (dy - pdy) ** 2)

                    if dist > self.threshold:
                        filtered_vectors.append((pdx, pdy))
                    else:
                        filtered_vectors.append((dx, dy))
            else:
                filtered_vectors = vectors

            # aggiorno stato
            self.prev_vectors = filtered_vectors

            # ricostruisco motion globale
            if len(filtered_vectors) > 0:
                mean = np.mean(filtered_vectors, axis=0)
                dx, dy = float(mean[0]), float(mean[1])
                magnitude = float(np.linalg.norm(mean))
                angle = float(np.arctan2(dy, dx))
            else:
                dx = dy = magnitude = angle = None

            output.append(
                SparseOpticalFlowSignalSample(
                    frame_index=sample.frame_index,
                    box=sample.box,
                    points=sample.points,
                    point_vectors=filtered_vectors,
                    motion_vector=(dx, dy) if dx is not None else None,
                    motion_magnitude=magnitude,
                    motion_angle=angle,
                    timestamp_seconds=sample.timestamp_seconds,
                    metadata=dict(sample.metadata),
                )
            )

        return Signal(output, config=dict(signal.config))

    def reset(self):
        self.prev_vectors = None
