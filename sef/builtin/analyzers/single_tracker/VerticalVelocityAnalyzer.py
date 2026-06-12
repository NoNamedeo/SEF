from __future__ import annotations

from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.ISignal import ISignal
from library.core.artifacts.data.TwoDimGraphData import TwoDimGraphData


class VerticalVelocityAnalyzer(IAnalyzer):
    """Compute the time-derivative (velocity) of vertical position."""

    def __init__(self, config=None):
        super().__init__(config)
        self.use_timestamps = bool(self.config.get("use_timestamps", True))

    def analyze(self, signal: ISignal) -> IData:
        y_values: list[float] = []
        t_values: list[float] = []

        for sample in signal:
            if sample.centroid is None:
                continue

            t = sample.timestamp_seconds if self.use_timestamps and sample.timestamp_seconds is not None else float(sample.frame_index)

            y_values.append(float(sample.centroid[1]))
            t_values.append(float(t))

        if len(y_values) < 2:
            raise ValueError("Not enough data to compute derivative")

        dy_dt: list[float] = []
        t_mid: list[float] = []

        for i in range(1, len(y_values) - 1):
            dy = y_values[i + 1] - y_values[i - 1]
            dt = t_values[i + 1] - t_values[i - 1]

            if dt == 0:
                continue

            dy_dt.append(dy / dt)
            t_mid.append(t_values[i])

        total_dy = y_values[-1] - y_values[0]
        total_dt = t_values[-1] - t_values[0]
        avg_velocity = total_dy / total_dt if total_dt != 0 else None

        x_label = "Time [s]" if self.use_timestamps else "Frame Index"
        y_label = "Velocity [px/s]" if self.use_timestamps else "Velocity [px/frame]"

        return TwoDimGraphData(
            x=t_mid,
            y=dy_dt,
            label="Vertical Velocity",
            title="Vertical Velocity Over Time",
            x_label=x_label,
            y_label=y_label,
            metadata={"points": len(dy_dt), "average_velocity": avg_velocity},
        )
