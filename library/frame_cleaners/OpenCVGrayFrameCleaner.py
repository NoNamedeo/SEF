from __future__ import annotations

from typing import Any

import cv2

from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame


class OpenCVGrayFrameCleaner(IFrameCleaner):
    """Convert a frame to grayscale while preserving frame metadata."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

    def clean(self, frame: Frame) -> Frame:
        gray_image = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2GRAY)
        return Frame(
            image=gray_image,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata=dict(frame.metadata),
        )
