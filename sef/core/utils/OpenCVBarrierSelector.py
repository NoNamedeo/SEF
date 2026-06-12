from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

from sef.core.utils.OpenCVDisplayUtils import DisplayTransform

Barrier = tuple[tuple[int, int], tuple[int, int]]


class OpenCVBarrierSelector:
    """Interactive utility to draw named barriers on the first frame of a video."""

    WINDOW_NAME = "Select Barriers"

    def __init__(self):
        self._base_frame = None
        self._current_start: tuple[int, int] | None = None
        self._cursor: tuple[int, int] | None = None
        self._display_transform: DisplayTransform | None = None

    def _mouse_callback(self, event, x, y, flags, param):
        if self._display_transform is None:
            return

        point = self._display_transform.to_original_point(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_start = point
            self._cursor = point
        elif event == cv2.EVENT_MOUSEMOVE and self._current_start is not None:
            self._cursor = point
        elif event == cv2.EVENT_LBUTTONUP and self._current_start is not None:
            self._cursor = point

    @classmethod
    def select_barriers(
        cls,
        video_path: str | Path,
        barrier_names: Iterable[str],
        resize: tuple[int, int] | None = None,
    ) -> dict[str, Barrier]:
        selector = cls()
        return selector._select(video_path, list(barrier_names), resize)

    def _select(
        self,
        video_path: str | Path,
        barrier_names: list[str],
        resize: tuple[int, int] | None,
    ) -> dict[str, Barrier]:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            success, frame = cap.read()
            if not success:
                raise ValueError("Cannot read first frame")
        finally:
            cap.release()

        if resize is not None:
            frame = cv2.resize(frame, resize)

        self._base_frame = frame
        self._display_transform = DisplayTransform.from_image(frame)

        barriers: dict[str, Barrier] = {}

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.WINDOW_NAME,
            self._display_transform.display_width,
            self._display_transform.display_height,
        )
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        barrier_index = 0
        try:
            while barrier_index < len(barrier_names):
                barrier_name = barrier_names[barrier_index]
                render = self._base_frame.copy()

                for index, (name, ((x1, y1), (x2, y2))) in enumerate(barriers.items(), start=1):
                    cv2.line(render, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        render,
                        f"{index}. {name}",
                        (x1 + 5, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )

                if self._current_start is not None and self._cursor is not None:
                    cv2.line(render, self._current_start, self._cursor, (0, 200, 0), 2)

                cv2.putText(
                    render,
                    f"Draw barrier: {barrier_name}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    render,
                    "Drag with mouse. SPACE/ENTER confirm, R reset current, ESC abort.",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                display_render = self._display_transform.resize_for_display(render)
                cv2.imshow(self.WINDOW_NAME, display_render)
                key = cv2.waitKey(20) & 0xFF

                if key in (13, 32):
                    if self._current_start is None or self._cursor is None:
                        continue
                    if self._current_start == self._cursor:
                        continue

                    barriers[barrier_name] = (self._current_start, self._cursor)
                    self._current_start = None
                    self._cursor = None
                    barrier_index += 1
                elif key in (ord("r"), ord("R")):
                    self._current_start = None
                    self._cursor = None
                elif key == 27:
                    raise ValueError("Barrier selection aborted")

            return barriers
        finally:
            cv2.destroyWindow(self.WINDOW_NAME)
