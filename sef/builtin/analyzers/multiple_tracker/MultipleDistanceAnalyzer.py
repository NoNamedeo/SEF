from __future__ import annotations

from typing import Tuple

from sef.core.artifacts.signal_sample.BoxSignalSample import Point2D
from sef.core.artifacts.data.TwoDimGraphData import TwoDimGraphData
from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.IData import IData
from sef.core.interfaces.ISignal import ISignal


class MultipleDistanceAnalyzer(IAnalyzer):
    """Build a distance series between two tracked objects."""

    def __init__(self, analyzed_pair: Tuple[int, int] = (0, 1), config=None):
        super().__init__(config)
        self.analyzed_pair = analyzed_pair
        self.use_timestamps = bool(self.config.get("use_timestamps", True))

    def analyze(self, signal: ISignal) -> IData:
        x_values: list[float] = []
        y_values: list[float] = []

        for multi_sample in signal:
            sample1 = multi_sample.samples.get(self.analyzed_pair[0])
            sample2 = multi_sample.samples.get(self.analyzed_pair[1])

            if sample1 is None or sample2 is None:
                continue

            if sample1.centroid is None or sample2.centroid is None:
                continue

            x_axis_value = (
                sample1.timestamp_seconds
                if self.use_timestamps and sample1.timestamp_seconds is not None
                else float(sample1.frame_index)
            )

            x_values.append(float(x_axis_value))

            y_values.append(self._distance(sample2.centroid, sample1.centroid))

        if not x_values:
            raise ValueError("Signal does not contain valid centroid data")

        x_label = "Time [s]" if self.use_timestamps else "Frame Index"

        return TwoDimGraphData(
            x=x_values,
            y=y_values,
            label="Distance",
            title="Distance Over Time",
            x_label=x_label,
            y_label="Distance [px]",
            metadata={"points": len(x_values)},
        )

    @staticmethod
    def _distance(point1: Point2D, point2: Point2D) -> float:
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5