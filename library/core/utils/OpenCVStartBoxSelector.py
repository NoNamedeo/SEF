import cv2
from typing import Any, Tuple

class OpenCVStartBoxSelector:
    """Utility class for finding the start box of a video"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @staticmethod
    def select_start(video_path: str, resize: Tuple[int, int] = None) -> Tuple[int, int, int, int]:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Cannot read first frame")

            if resize is not None:
                frame = cv2.resize(frame, resize)

            roi = cv2.selectROI(
                "Select Start Box",
                frame,
                fromCenter=False,
                showCrosshair=True
            )

            cv2.destroyWindow("Select Start Box")

            x, y, w, h = map(int, roi)

            if w == 0 or h == 0:
                raise ValueError("No ROI selected")

            return x, y, w, h
        finally:
            cap.release()
