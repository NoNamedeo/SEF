import cv2
from typing import Any, Tuple, List

from sef.core.artifacts.Frame import Frame
from sef.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from sef.core.utils.OpenCVDisplayUtils import DisplayTransform


class OpenCVMultiStartBoxSelector:
    """Utility class for selecting multiple start boxes from a video."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @staticmethod
    def select_start(
        video_path: str,
        num_boxes: int = 3,
        resize: Tuple[int, int] | None = None,
        apply_preprocessing: bool = True,
        single_frame_processors: list[ISingleFrameProcessor] | None = None,
    ) -> List[Tuple[int, int, int, int]]:

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
                fake_frame = Frame(
                    image=frame,
                    index=0,
                    timestamp_seconds=0,
                    metadata={}
                )

                for processor in single_frame_processors:
                    fake_frame = processor.process(fake_frame)

                frame = fake_frame.frame

            transform = DisplayTransform.from_image(frame)
            boxes: List[Tuple[int, int, int, int]] = []
            window_name = "Select Start Boxes"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, transform.display_width, transform.display_height)

            for i in range(num_boxes):
                display = frame.copy()

                for (x, y, w, h) in boxes:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(
                    display,
                    f"Select ROI {i+1}/{num_boxes} (ESC to stop)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                display_for_selection = transform.resize_for_display(display)
                roi = cv2.selectROI(
                    window_name,
                    display_for_selection,
                    fromCenter=False,
                    showCrosshair=True,
                )

                x, y, w, h = transform.to_original_roi(tuple(map(int, roi)))

                if w == 0 or h == 0:
                    break

                boxes.append((x, y, w, h))

            cv2.destroyWindow(window_name)

            if not boxes:
                raise ValueError("No ROI selected")

            return boxes

        finally:
            cap.release()
