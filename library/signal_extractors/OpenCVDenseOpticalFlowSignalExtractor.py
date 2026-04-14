from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from library.core.artifacts.DenseOpticalFlowSignalSample import (
    DenseOpticalFlowSignalSample,
)
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor


class OpenCVDenseFarnebackSignalExtractor(ISignalExtractor):
    """
    Dense optical flow extractor using Farneback algorithm.
    Produces a grid-based motion field instead of sparse points.
    """

    def __init__(
        self,
        cell_size: int = 16,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        self.cell_size = cell_size

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples: list[DenseOpticalFlowSignalSample] = []

        prev_gray = None

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position
            # TODO: da rendere un opzione
            gray = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                prev_gray = gray

                samples.append(
                    DenseOpticalFlowSignalSample(
                        frame_index=frame_index,
                        grid_shape=(0, 0),
                        cell_size=self.cell_size,
                        motion_field=[],
                        motion_vector=None,
                        motion_magnitude=None,
                        motion_angle=None,
                        timestamp_seconds=frame.timestamp_seconds,
                        metadata=dict(frame.metadata),
                    )
                )
                continue

            # algoritmo farneback dense flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            # costruisco la griglia
            rows = flow.shape[0] // self.cell_size
            cols = flow.shape[1] // self.cell_size

            motion_field: list[tuple[float, float]] = []

            for r in range(rows):
                for c in range(cols):
                    # faccio la media dei vettori contenuti nella cella
                    y0 = r * self.cell_size
                    x0 = c * self.cell_size

                    cell = flow[y0 : y0 + self.cell_size, x0 : x0 + self.cell_size]

                    if cell.size == 0:
                        motion_field.append((0.0, 0.0))
                        continue

                    mean_flow = np.mean(cell.reshape(-1, 2), axis=0)
                    dx, dy = float(mean_flow[0]), float(mean_flow[1])

                    motion_field.append((dx, dy))

            # vettore di movimento globale
            if len(motion_field) > 0:
                mean_vec = np.mean(np.array(motion_field), axis=0)
                dx, dy = float(mean_vec[0]), float(mean_vec[1])

                magnitude = float(np.linalg.norm(mean_vec))
                angle = float(np.arctan2(dy, dx))
                global_motion = (dx, dy)
            else:
                global_motion = None
                magnitude = None
                angle = None

            samples.append(
                DenseOpticalFlowSignalSample(
                    frame_index=frame_index,
                    grid_shape=(rows, cols),
                    cell_size=self.cell_size,
                    motion_field=motion_field,
                    motion_vector=global_motion,
                    motion_magnitude=magnitude,
                    motion_angle=angle,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata={
                        **dict(frame.metadata),
                        "method": "farneback",
                        "dense": True,
                    },
                )
            )

            prev_gray = gray

        return Signal(samples)

    def track(self, buffer: FrameBuffer) -> ISignal:
        return self.extract(buffer)
