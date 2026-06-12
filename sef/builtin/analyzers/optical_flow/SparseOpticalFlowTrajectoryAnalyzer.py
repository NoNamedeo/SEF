from __future__ import annotations

from typing import Any

from sef.core.artifacts.data.TrajectoryData import TrajectoryData
from sef.core.artifacts.signal_sample.SparseOpticalFlowSignalSample import SparseOpticalFlowSignalSample
from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.ISignal import ISignal


class SparseOpticalFlowTrajectoryAnalyzer(IAnalyzer):
    """
    Builds point trajectories from sparse optical flow samples.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.max_tracks = self.config.get("max_tracks", None)

    def analyze(self, signal: ISignal) -> TrajectoryData:
        trajectories_x: list[list[float]] = []
        trajectories_y: list[list[float]] = []

        frame_indices: list[int] = []
        timestamps: list[float] = []

        initialized = False

        for sample in signal:
            if not isinstance(sample, SparseOpticalFlowSignalSample):
                continue

            if not sample.points:
                continue

            # init trajectories on first valid frame
            if not initialized:
                n_tracks = len(sample.points)

                if self.max_tracks is not None:
                    n_tracks = min(n_tracks, self.max_tracks)

                trajectories_x = [[] for _ in range(n_tracks)]
                trajectories_y = [[] for _ in range(n_tracks)]
                initialized = True

            n_tracks = min(len(sample.points), len(trajectories_x))

            for i in range(n_tracks):
                x, y = sample.points[i]
                vx, vy = sample.point_vectors[i] if i < len(sample.point_vectors) else (0.0, 0.0)

                # update position using velocity (trajectory reconstruction)
                if len(trajectories_x[i]) == 0:
                    trajectories_x[i].append(x)
                    trajectories_y[i].append(y)
                else:
                    prev_x = trajectories_x[i][-1]
                    prev_y = trajectories_y[i][-1]

                    trajectories_x[i].append(prev_x + vx)
                    trajectories_y[i].append(prev_y + vy)

            frame_indices.append(sample.frame_index)
            if sample.timestamp_seconds is not None:
                timestamps.append(sample.timestamp_seconds)

        if not trajectories_x:
            raise ValueError("No valid trajectory data found in signal")

        return TrajectoryData(
            trajectories_x=trajectories_x,
            trajectories_y=trajectories_y,
            frame_indices=frame_indices,
            timestamps=timestamps,
            metadata={
                "tracks": len(trajectories_x),
                "type": "optical_flow_trajectories",
            },
        )
