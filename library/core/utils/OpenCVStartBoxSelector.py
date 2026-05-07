import cv2
from typing import Any, Tuple

from library.core.artifacts.Frame import Frame
from library.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor


class OpenCVStartBoxSelector:
    """Utility class for finding the start box of a video with optional preprocessing."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @staticmethod
    def select_start(
        video_path: str,
        resize: Tuple[int, int] | None = None,
        apply_preprocessing: bool = True,
        single_frame_processors: list[ISingleFrameProcessor] | None = None,
    ) -> Tuple[int, int, int, int]:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Cannot read first frame")

            if resize is not None:
                frame = cv2.resize(frame, resize)

            if apply_preprocessing and single_frame_processors:
                temp_frame = Frame(
                    image=frame,
                    index=0,
                    timestamp_seconds=0,
                    metadata={}
                )

                for processor in single_frame_processors:
                    temp_frame = processor.process(temp_frame)

                frame = temp_frame.frame

            roi = cv2.selectROI(
                "Select Start Box",
                frame,
                fromCenter=False,
                showCrosshair=True,
            )

            cv2.destroyWindow("Select Start Box")

            x, y, w, h = map(int, roi)

            if w == 0 or h == 0:
                raise ValueError("No ROI selected")

            return x, y, w, h

        finally:
            cap.release()