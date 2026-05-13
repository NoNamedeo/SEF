from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.IFrameExtractor import IFrameExtractor


class OpenCVBufferedFrameExtractor(IFrameExtractor):
    """Read a video with OpenCV and expose raw frames through a FrameBuffer."""

    DEFAULT_MAX_FRAMES = 300
    HARD_MAX_FRAMES = 10_000

    def __init__(
        self,
        path: str | Path,
        buffer: FrameBuffer | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.path = str(path)
        self.buffer = buffer or FrameBuffer()
        self.resize = self.config.get("resize")
        self.stride = int(self.config.get("stride", 1))
        self.max_frames = int(self.config.get("max_frames", self.DEFAULT_MAX_FRAMES))

        if self.stride <= 0:
            raise ValueError("stride must be greater than 0")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be greater than 0")
        if self.max_frames > self.HARD_MAX_FRAMES:
            raise ValueError(f"max_frames cannot exceed hard limit {self.HARD_MAX_FRAMES}")

    def extract(self) -> FrameBuffer:
        buffer = self.buffer
        cap = cv2.VideoCapture(self.path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        source_frame_index = -1
        yielded_frames = 0

        try:
            while True:
                success, image = cap.read()
                if not success:
                    break

                source_frame_index += 1
                if source_frame_index % self.stride != 0:
                    continue

                if self.resize is not None:
                    image = cv2.resize(image, self.resize)

                timestamp_seconds = (source_frame_index / fps) if fps > 0 else None
                frame_metadata = {
                    "source_path": self.path,
                    "frame_size": (int(image.shape[1]), int(image.shape[0])),
                }
                if self.resize is not None:
                    frame_metadata["resize"] = tuple(int(value) for value in self.resize)
                if fps > 0:
                    frame_metadata["source_fps"] = fps
                frame = Frame(
                    image=image,
                    index=source_frame_index,
                    timestamp_seconds=timestamp_seconds,
                    metadata=frame_metadata,
                )
                buffer.put(frame)
                yielded_frames += 1

                if self.max_frames is not None and yielded_frames >= self.max_frames:
                    break
        finally:
            cap.release()
            buffer.close()

        return buffer
