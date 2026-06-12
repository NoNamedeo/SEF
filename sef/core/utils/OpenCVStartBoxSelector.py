import cv2
from typing import Any, Tuple

from sef.core.artifacts.Frame import Frame
from sef.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from sef.core.utils.OpenCVDisplayUtils import DisplayTransform


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

            transform = DisplayTransform.from_image(frame)
            display_frame = transform.resize_for_display(frame)

            window_name = "Select Start Box"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, transform.display_width, transform.display_height)

            roi = cv2.selectROI(
                window_name,
                display_frame,
                fromCenter=False,
                showCrosshair=True,
            )

            cv2.destroyWindow(window_name)

            x, y, w, h = transform.to_original_roi(tuple(map(int, roi)))

            if w == 0 or h == 0:
                raise ValueError("No ROI selected")

            return x, y, w, h

        finally:
            cap.release()
