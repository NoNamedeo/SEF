from __future__ import annotations

from collections.abc import Iterable

from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.artifacts.TwoDimPointData import TwoDimPointData
from library.core.interfaces.ISignalSample import ISignalSample
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import IStreamingAnalyzer


class VerticalPositionStreamAnalyzer(IStreamingAnalyzer):
    """Build a y-position series from extracted centroids."""

    capabilities = StageCapabilities.streaming(
        stateful=False,
        preserves_order=True,
        realtime_safe=True,
    )

    def __init__(self, buffer: DataBuffer = None, config=None):
        super().__init__(config)
        self._default_buffer = buffer
        self.use_timestamps = bool(self.config.get("use_timestamps", True))

    def analyze(self, signal: Iterable[ISignalSample], consumer_id: int = 0) -> TwoDimGraphData:
        output = self._default_buffer or DataBuffer()
        return self.analyze_into(signal, output, consumer_id=consumer_id)

    def analyze_into(
        self,
        signal: Iterable[ISignalSample],
        output_buffer: DataBuffer,
        consumer_id: int = 0,
    ) -> TwoDimGraphData:
        x_label = "Time [s]" if self.use_timestamps else "Frame Index"
        x_values: list[float] = []
        y_values: list[float] = []

        for sample in signal:
            if sample.centroid is None:
                continue

            x_value = (
                float(sample.timestamp_seconds)
                if self.use_timestamps and sample.timestamp_seconds is not None
                else float(sample.frame_index)
            )

            y_value = float(-sample.centroid[1])
            x_values.append(x_value)
            y_values.append(y_value)

            output_buffer.put(
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

        output_buffer.close()
        return TwoDimGraphData(
            x=x_values,
            y=y_values,
            label="Vertical Position",
            title="Vertical Position Over Time",
            x_label=x_label,
            y_label="Y Position [px]",
            metadata={"points": len(x_values), "consumer_id": consumer_id},
        )
