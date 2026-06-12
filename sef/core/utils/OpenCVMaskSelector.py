from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

from library.core.artifacts.Frame import Frame
from library.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from library.core.utils.OpenCVDisplayUtils import DisplayTransform


class OpenCVMaskSelector:
    """
    Utility class for manually drawing a mask on the first frame of a video.

    Controls:
        - Hold left mouse button to draw in brush mode
        - Hold left mouse button to select a rectangle in box mode
        - Press B to switch to brush mode
        - Press R to switch to box mode
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
            transform = DisplayTransform.from_image(original)

            mask = np.zeros(original.shape[:2], dtype=np.uint8)

            drawing = False
            mode = "brush"
            rectangle_start: tuple[int, int] | None = None
            rectangle_end: tuple[int, int] | None = None

            window_name = "Draw Mask"
            display_brush_radius = max(int(round(brush_radius * transform.scale)), 1)

            def mouse_callback(event, x, y, flags, param):
                nonlocal drawing, mask, mode, rectangle_start, rectangle_end
                original_x, original_y = transform.to_original_point(x, y)

                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    if mode == "brush":
                        cv2.circle(
                            mask,
                            (original_x, original_y),
                            brush_radius,
                            255,
                            -1,
                        )
                    else:
                        rectangle_start = (original_x, original_y)
                        rectangle_end = (original_x, original_y)

                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        if mode == "brush":
                            cv2.circle(
                                mask,
                                (original_x, original_y),
                                brush_radius,
                                255,
                                -1,
                            )
                        else:
                            rectangle_end = (original_x, original_y)

                elif event == cv2.EVENT_LBUTTONUP:
                    if mode == "brush":
                        cv2.circle(
                            mask,
                            (original_x, original_y),
                            brush_radius,
                            255,
                            -1,
                        )
                    elif rectangle_start is not None:
                        rectangle_end = (original_x, original_y)
                        x1 = min(rectangle_start[0], rectangle_end[0])
                        y1 = min(rectangle_start[1], rectangle_end[1])
                        x2 = max(rectangle_start[0], rectangle_end[0])
                        y2 = max(rectangle_start[1], rectangle_end[1])
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        rectangle_start = None
                        rectangle_end = None
                    drawing = False

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, transform.display_width, transform.display_height)

            cv2.setMouseCallback(window_name, mouse_callback)

            while True:
                overlay = original.copy()
                overlay[mask > 0] = (0, 0, 255)

                if mode == "box" and drawing and rectangle_start is not None and rectangle_end is not None:
                    x1 = min(rectangle_start[0], rectangle_end[0])
                    y1 = min(rectangle_start[1], rectangle_end[1])
                    x2 = max(rectangle_start[0], rectangle_end[0])
                    y2 = max(rectangle_start[1], rectangle_end[1])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

                cv2.putText(
                    overlay,
                    "Mask: B brush | R box | ENTER/SPACE confirm | C clear | ESC cancel",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                display_overlay = transform.resize_for_display(overlay)
                cv2.putText(
                    display_overlay,
                    f"Mode: {mode} | Brush: {display_brush_radius}px",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(window_name, display_overlay)

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
                    drawing = False
                    rectangle_start = None
                    rectangle_end = None
                elif key in (ord("b"), ord("B")):
                    mode = "brush"
                    drawing = False
                    rectangle_start = None
                    rectangle_end = None
                elif key in (ord("r"), ord("R")):
                    mode = "box"
                    drawing = False
                    rectangle_start = None
                    rectangle_end = None

            cv2.destroyWindow(window_name)

            return mask

        finally:
            cap.release()
