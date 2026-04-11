from __future__ import annotations

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.IData import IData
from library.core.abstractions.ISignal import ISignal
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData


class HorizontalVelocityAnalyzer(IAnalyzer):
    """Compute the time-derivative (velocity) of horizontal position."""

    def __init__(self, config=None):
        super().__init__(config)
        self.use_timestamps = bool(self.config.get("use_timestamps", True))

    def analyze(self, signal: ISignal) -> IData:
        x_values: list[float] = []
        t_values: list[float] = []

        for sample in signal:
            if sample.centroid is None:
                continue

            t = (
                sample.timestamp_seconds
                if self.use_timestamps and sample.timestamp_seconds is not None
                else float(sample.frame_index)
            )

            x_values.append(float(sample.centroid[0]))
            t_values.append(float(t))

        if len(x_values) < 2:
            raise ValueError("Not enough data to compute derivative")

        dx_dt: list[float] = []
        t_mid: list[float] = []

        for i in range(1, len(x_values) - 1):
            dx = x_values[i + 1] - x_values[i - 1]
            dt = t_values[i + 1] - t_values[i - 1]

            if dt == 0:
                continue

            dx_dt.append(dx / dt)
            t_mid.append(t_values[i])

        total_dx = x_values[-1] - x_values[0]
        total_dt = t_values[-1] - t_values[0]
        avg_velocity = total_dx / total_dt if total_dt != 0 else None

        x_label = "Time [s]" if self.use_timestamps else "Frame Index"
        y_label = "Velocity [px/s]" if self.use_timestamps else "Velocity [px/frame]"

        return TwoDimGraphData(
            x=t_mid,
            y=dx_dt,
            label="Horizontal Velocity",
            title="Horizontal Velocity Over Time",
            x_label=x_label,
            y_label=y_label,
            metadata={"points": len(dx_dt),
                      "average_velocity": avg_velocity},
        )