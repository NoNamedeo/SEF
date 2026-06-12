from __future__ import annotations

from typing import Any

import cv2

from sef.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from sef.core.artifacts.Frame import Frame
from sef.core.artifacts.signal_sample.BoxSignalSample import BoundingBox


class OpenCVZoomFrameProcessor(ISingleFrameProcessor):
    """Zoom into a fixed ROI (crop + upscale back to original frame size)."""

    def __init__(
        self,
        roi: BoundingBox,
        interpolation: int = cv2.INTER_LINEAR,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.roi = roi
        self.interpolation = interpolation

        x, y, w, h = self.roi
        if w <= 0 or h <= 0:
            raise ValueError("ROI must have positive width and height")

    def process(self, frame: Frame) -> Frame:
        x, y, w, h = self.roi

        img = frame.frame
        H, W = img.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)

        cropped = img[y1:y2, x1:x2]

        if cropped.size == 0:
            raise ValueError("ROI resulted in empty crop")

        zoomed = cv2.resize(
            cropped,
            (W, H),
            interpolation=self.interpolation,
        )

        return Frame(
            image=zoomed,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata=dict(frame.metadata),
        )