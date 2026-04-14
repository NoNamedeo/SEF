from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable

import cv2

from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer


class OpenCVBufferedFrameExtractor(IFrameExtractor):
    """Read a video with OpenCV and expose cleaned frames through a FrameBuffer."""

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
        self.max_frames = self.config.get("max_frames")

        if self.stride <= 0:
            raise ValueError("stride must be greater than 0")

    def extract(self, frame_cleaners: Iterable[IFrameCleaner]) -> FrameBuffer:
        buffer = self.buffer.clone_empty()
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
                frame = Frame(
                    image=image,
                    index=source_frame_index,
                    timestamp_seconds=timestamp_seconds,
                    metadata={"source_path": self.path},
                )

                cleaned_frame = frame

                for cleaner in frame_cleaners:
                    cleaned_frame = cleaner.clean(frame)

                buffer.put(cleaned_frame)
                yielded_frames += 1

                if self.max_frames is not None and yielded_frames >= self.max_frames:
                    break
        finally:
            cap.release()
            buffer.close()

        return buffer
