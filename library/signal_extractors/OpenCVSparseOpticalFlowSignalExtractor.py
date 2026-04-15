from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.SparseOpticalFlowSignalSample import (
    SparseOpticalFlowSignalSample,
    BoundingBox,
)


class OpenCVSparseOpticalFlowSignalExtractor(ISignalExtractor):
    """
    Sparse optical flow (Lucas-Kanade) with per-point trajectories.
    """

    def __init__(
        self,
        example_box: BoundingBox | None = None,
        max_corners: int = 100,
        quality_level: float = 0.3,
        min_distance: int = 7,
        block_size: int = 7,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        self.example_box = example_box

        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples: list[SparseOpticalFlowSignalSample] = []

        prev_gray = None
        prev_points = None

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position

            # TODO va fatto diventare una configurazione
            gray = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2GRAY)

            if position == 0:
                mask = None

                if self.example_box is not None:
                    x, y, w, h = self.example_box
                    mask = np.zeros_like(gray)
                    mask[y : y + h, x : x + w] = 255

                # traccia i "punti buoni" ispirandosi, se c'è, a quelli della example_box
                # TODO sarebbe da fare una classe/sistema a parte per tracciare punti piu specifici
                # (tipo allenato per riconoscere articolazioni braccio)
                prev_points = cv2.goodFeaturesToTrack(
                    gray,
                    mask=mask,
                    **self.feature_params,
                )

                prev_gray = gray

                samples.append(
                    SparseOpticalFlowSignalSample(
                        frame_index=frame_index,
                        box=self.example_box,
                        points=([tuple(p.ravel()) for p in prev_points] if prev_points is not None else []),
                        point_vectors=[],
                        motion_vector=None,
                        motion_magnitude=None,
                        motion_angle=None,
                        timestamp_seconds=frame.timestamp_seconds,
                        metadata=dict(frame.metadata),
                    )
                )
                continue

            if prev_points is None or len(prev_points) == 0:
                samples.append(
                    SparseOpticalFlowSignalSample(
                        frame_index=frame_index,
                        box=self.example_box,
                        points=[],
                        point_vectors=[],
                        motion_vector=None,
                        motion_magnitude=None,
                        motion_angle=None,
                        timestamp_seconds=frame.timestamp_seconds,
                        metadata=dict(frame.metadata),
                    )
                )

                prev_gray = gray
                continue

            # algoritmo di sparse optical flow
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_points,
                None,
                **self.lk_params,
            )

            # filter valid points
            status = status.reshape(-1)
            good_new = next_points[status == 1]
            good_old = prev_points[status == 1]

            if len(good_new) > 0:
                # per-point vectors
                flow_vectors = good_new - good_old

                points = [tuple(p.ravel()) for p in good_new]
                point_vectors = [tuple(v.ravel()) for v in flow_vectors]

                # global motion
                mean_flow = np.mean(flow_vectors, axis=0)
                dx, dy = float(mean_flow[0]), float(mean_flow[1])

                magnitude = float(np.linalg.norm(mean_flow))
                angle = float(np.arctan2(dy, dx))

                motion_vector = (dx, dy)
            else:
                points = []
                point_vectors = []
                motion_vector = None
                magnitude = None
                angle = None

            if self.config.get("show") and len(points) > 0:
                for p_new, p_old in zip(good_new, good_old):
                    x_new, y_new = p_new.ravel()
                    x_old, y_old = p_old.ravel()

                    cv2.line(
                        frame.frame,
                        (int(x_old), int(y_old)),
                        (int(x_new), int(y_new)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(
                        frame.frame,
                        (int(x_new), int(y_new)),
                        3,
                        (0, 0, 255),
                        -1,
                    )

                cv2.imshow("Optical Flow", frame.frame)
                if cv2.waitKey(1) == 27:
                    break

            samples.append(
                SparseOpticalFlowSignalSample(
                    frame_index=frame_index,
                    box=self.example_box,
                    points=points,
                    point_vectors=point_vectors,
                    motion_vector=motion_vector,
                    motion_magnitude=magnitude,
                    motion_angle=angle,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata=dict(frame.metadata),
                )
            )

            prev_gray = gray
            prev_points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

        return Signal(samples)

    def track(self, buffer: FrameBuffer) -> ISignal:
        return self.extract(buffer)
