from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame


class OpenCVBackgroundSubtractionFrameCleaner(IFrameCleaner):
    """
    Applies background subtraction with strong noise filtering
    to isolate meaningful moving objects.
    """

    def __init__(
        self,
        method: str = "MOG2",
        detect_shadows: bool = False,
        kernel_size: int = 5,
        min_area: int = 500,
        apply_close: bool = True,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        self.method = method.upper()
        self.detect_shadows = detect_shadows

        self.kernel_size = kernel_size
        self.min_area = min_area
        self.apply_close = apply_close

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

        # background subtraction
        fg_mask = self._bg_model.apply(image)

        # remove shadows (MOG2 outputs 127 for shadows)
        fg_mask[fg_mask == 127] = 0

        # morphology filtering
        kernel = np.ones(
            (self.kernel_size, self.kernel_size),
            np.uint8,
        )

        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        if self.apply_close:
            fg_mask = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_CLOSE,
                kernel,
            )

        # apply mask
        cleaned = cv2.bitwise_and(
            image,
            image,
            mask=fg_mask,
        )

        return Frame(
            image=cleaned,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata={
                **dict(frame.metadata),
                "fg_mask_mean": float(np.mean(fg_mask)),
                "min_area": self.min_area,
                "kernel_size": self.kernel_size,
            },
        )