from __future__ import annotations

from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.artifacts.TwoDimPointData import TwoDimPointData
from library.core.interfaces.IAnalyzer import IAnalyzer


class VerticalPositionStreamAnalyzer(IAnalyzer):
    """Build a y-position series from extracted centroids."""

    def __init__(self, buffer: DataBuffer = None, config=None):
        super().__init__(config)
        self.buffer = buffer or DataBuffer()
        self.use_timestamps = bool(self.config.get("use_timestamps", True))

    def analyze(self, signal: SignalBuffer) -> DataBuffer:
        data_buffer = self.buffer.clone_empty()
        x_label = "Time [s]" if self.use_timestamps else "Frame Index"

        for sample in signal:
            if sample.centroid is None:
                continue

            x_value = float(sample.timestamp_seconds) if self.use_timestamps and sample.timestamp_seconds is not None else float(sample.frame_index)
            y_value = float(-sample.centroid[1])

            data_buffer.put(
                TwoDimPointData(
                    x=x_value,
                    y=y_value,
                    label="Vertical Position",
                    title="Vertical Position Over Time",
                    x_label=x_label,
                    y_label="Y Position [px]",
                    metadata={},
                )
            )

        data_buffer.close()
        return data_buffer
