from __future__ import annotations

from typing import Any

import cv2

from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame


class OpenCVHistogramEqualizationFrameCleaner(IFrameCleaner):
    """Apply histogram equalization to improve contrast."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

    def clean(self, frame: Frame) -> Frame:
        image = frame.frame

        if len(image.shape) == 2:
            eq = cv2.equalizeHist(image)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        return Frame(
            image=eq,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata=dict(frame.metadata),
        )