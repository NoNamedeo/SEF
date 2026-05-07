from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

from library.core.artifacts.Frame import Frame
from library.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor


class OpenCVMaskSelector:
    """
    Utility class for manually drawing a mask on the first frame of a video.

    Controls:
        - Hold left mouse button to draw
        - Press ENTER or SPACE to confirm
        - Press C to clear
        - Press ESC to cancel
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @staticmethod
    def select_mask(
        video_path: str,
        resize: Tuple[int, int] | None = None,
        apply_preprocessing: bool = True,
        single_frame_processors: list[ISingleFrameProcessor] | None = None,
        brush_radius: int = 10,
    ) -> np.ndarray:

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
                    metadata={},
                )

                for processor in single_frame_processors:
                    temp_frame = processor.process(temp_frame)

                frame = temp_frame.frame

            original = frame.copy()

            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            drawing = False

            window_name = "Draw Mask"

            def mouse_callback(event, x, y, flags, param):
                nonlocal drawing, frame, mask

                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True

                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:

                        cv2.circle(
                            mask,
                            (x, y),
                            brush_radius,
                            255,
                            -1,
                        )

                        cv2.circle(
                            frame,
                            (x, y),
                            brush_radius,
                            (0, 0, 255),
                            -1,
                        )

                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False

            cv2.namedWindow(window_name)

            cv2.setMouseCallback(window_name, mouse_callback)

            while True:

                overlay = frame.copy()

                cv2.putText(
                    overlay,
                    "Draw mask - ENTER/SPACE confirm | C clear | ESC cancel",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(window_name, overlay)

                key = cv2.waitKey(1) & 0xFF

                # ENTER or SPACE
                if key in (13, 32):

                    if np.count_nonzero(mask) == 0:
                        raise ValueError("Empty mask selected")

                    break

                # ESC
                elif key == 27:
                    raise ValueError("Mask selection cancelled")

                # C
                elif key in (ord("c"), ord("C")):
                    mask[:] = 0
                    frame = original.copy()

            cv2.destroyWindow(window_name)

            return mask

        finally:
            cap.release()