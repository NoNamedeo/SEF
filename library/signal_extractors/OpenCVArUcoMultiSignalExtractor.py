from __future__ import annotations

from typing import Any, Dict, List

import cv2

from library.core.artifacts.MultiManualSignalSample import MultiManualSignalSample
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.BoxSignalSample import BoundingBox, BoxSignalSample


class OpenCVArUcoMultiSignalExtractor(ISignalExtractor):
    """
    Detects ArUco markers in each frame and builds independent signals per marker ID.
    No tracking is used: detection is performed frame-by-frame.
    """

    def __init__(
        self,
        aruco_dict: int = cv2.aruco.DICT_4X4_50,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config or {})

        if not hasattr(cv2, "aruco"):
            raise ImportError("cv2.aruco is not available. Install opencv-contrib-python.")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.parameters = cv2.aruco.DetectorParameters()

        # OpenCV >= 4.7 uses ArucoDetector API
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples: List[MultiManualSignalSample] = []

        for position, frame in enumerate(buffer):
            frame_index = frame.index if frame.index is not None else position

            gray = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = self.detector.detectMarkers(gray)

            sample = MultiManualSignalSample(samples={})

            if ids is not None:
                ids = ids.flatten()

                for marker_corners, marker_id in zip(corners, ids):
                    pts = marker_corners.reshape(4, 2)

                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)

                    x, y = int(x_min), int(y_min)
                    w, h = int(x_max - x_min), int(y_max - y_min)

                    box: BoundingBox = (x, y, w, h)
                    centroid = (x + w / 2.0, y + h / 2.0)

                    sample.samples[int(marker_id)] = BoxSignalSample(
                        frame_index=frame_index,
                        box=box,
                        centroid=centroid,
                        timestamp_seconds=frame.timestamp_seconds,
                        metadata={
                            **frame.metadata,
                            "aruco_id": int(marker_id),
                        },
                    )

            samples.append(sample)

            if self.config.get("show"):
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame.frame, corners, ids)

                cv2.imshow("ArUco Detection", frame.frame)
                if cv2.waitKey(1) == 27:
                    break

        return Signal(samples)

    def track(self, buffer: FrameBuffer) -> ISignal:
        return self.extract(buffer)