from __future__ import annotations

from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.ISignal import ISignal


class HorizontalPositionAnalyzer(IAnalyzer):
    """Build a x-position series from extracted centroids."""

    def __init__(self, config=None):
        super().__init__(config)
        self.use_timestamps = bool(self.config.get("use_timestamps", True))

    def analyze(self, signal: ISignal) -> IData:
        x_values: list[float] = []
        y_values: list[float] = []

        for sample in signal:
            if sample.centroid is None:
                continue

            x_axis_value = sample.timestamp_seconds if self.use_timestamps and sample.timestamp_seconds is not None else float(sample.frame_index)
            x_values.append(float(x_axis_value))
            y_values.append(float(sample.centroid[0]))

        if not x_values:
            raise ValueError("Signal does not contain valid centroid data")

        x_label = "Time [s]" if self.use_timestamps else "Frame Index"

        return TwoDimGraphData(
            x=x_values,
            y=y_values,
            label="Horizontal Position",
            title="Horizontal Position Over Time",
            x_label=x_label,
            y_label="X Position [px]",
            metadata={"points": len(x_values)},
        )
