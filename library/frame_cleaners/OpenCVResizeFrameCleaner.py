from __future__ import annotations

from typing import Any, Tuple

import cv2

from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame


class OpenCVResizeFrameCleaner(IFrameCleaner):
    """Resize frames to a fixed resolution while preserving metadata."""

    def __init__(
        self,
        size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.size = size
        self.interpolation = interpolation

        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError("size must contain positive width and height")

    def clean(self, frame: Frame) -> Frame:
        resized_image = cv2.resize(
            frame.frame,
            self.size,
            interpolation=self.interpolation,
        )

        return Frame(
            image=resized_image,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata=dict(frame.metadata),
        )
