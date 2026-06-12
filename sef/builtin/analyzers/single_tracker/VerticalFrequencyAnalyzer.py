from __future__ import annotations

import numpy as np

from sef.core.artifacts.data.TwoDimGraphData import TwoDimGraphData
from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.IData import IData
from sef.core.interfaces.ISignal import ISignal


class VerticalFrequencyAnalyzer(IAnalyzer):
    """Build a frequency graph from extracted centroids."""

    def __init__(self, config=None):
        super().__init__(config)

    def analyze(self, signal: ISignal) -> IData:
        y_values: list[float] = []
        timestamps: list[float] = []

        for sample in signal:
            if sample.centroid is None:
                continue
            y_values.append(float(sample.centroid[1]))
            timestamps.append(float(sample.timestamp_seconds))

        dt = np.median(np.diff(np.array(timestamps)))

        y_positions = np.array(y_values) - np.mean(y_values)
        size = len(y_positions)
        frequency = np.fft.rfftfreq(size, d=dt)
        amplitudes = np.fft.rfft(y_positions)

        mask = frequency >= 0
        positive_frequency = frequency[mask]
        positive_amplitudes = amplitudes[mask]

        normalized_positive_amplitudes = (1.0 / size) * np.abs(positive_amplitudes)
        normalized_positive_amplitudes[1:-1] *= 2

        idx_max = np.argmax(normalized_positive_amplitudes)
        max_freq = positive_frequency[idx_max]
        max_amp = normalized_positive_amplitudes[idx_max]

        return TwoDimGraphData(
            x=positive_frequency.tolist(),
            y=normalized_positive_amplitudes.tolist(),
            label="Vertical Frequency Spectrum",
            title="Vertical Frequency Spectrum",
            x_label="Frequency [Hz]",
            y_label="Amplitude",
            metadata={"points": len(normalized_positive_amplitudes), "max_frequency": max_freq, "max_amplitude": max_amp},
        )
