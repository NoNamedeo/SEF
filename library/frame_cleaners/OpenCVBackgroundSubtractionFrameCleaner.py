from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame


class OpenCVBackgroundSubtractionFrameCleaner(IFrameCleaner):
    """
    Applies background subtraction to isolate moving objects.
    NOTE: This cleaner is stateful (keeps background model internally).
    """

    def __init__(
        self,
        method: str = "MOG2",
        detect_shadows: bool = False,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.method = method.upper()
        self.detect_shadows = detect_shadows

        self._bg_model = self._create_model()

    def _create_model(self):
        if self.method == "MOG2":
            return cv2.createBackgroundSubtractorMOG2(
                detectShadows=self.detect_shadows
            )
        elif self.method == "KNN":
            return cv2.createBackgroundSubtractorKNN()
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def clean(self, frame: Frame) -> Frame:
        image = frame.frame

        fg_mask = self._bg_model.apply(image)

        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        if len(image.shape) == 2:
            cleaned = cv2.bitwise_and(image, image, mask=fg_mask)
        else:
            cleaned = cv2.bitwise_and(image, image, mask=fg_mask)

        return Frame(
            image=cleaned,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata={
                **dict(frame.metadata),
                "fg_mask_mean": float(np.mean(fg_mask)),
            },
        )