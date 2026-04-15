from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame


class OpenCVBackgroundSubtractionFrameCleaner(IFrameCleaner):
    """
    Applies background subtraction to isolate moving objects.
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
            return cv2.createBackgroundSubtractorMOG2(detectShadows=self.detect_shadows)
        elif self.method == "KNN":
            return cv2.createBackgroundSubtractorKNN()
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def clean(self, frame: Frame) -> Frame:
        image = frame.frame

        # restituisce una maschera che distingue i movimenti (foreground, valore 255) dallo
        # sfondo (background, valore 0)
        # se detectShadows == True, traccia anche le "ombre" (valore 127)
        fg_mask = self._bg_model.apply(image)

        kernel = np.ones((3, 3), np.uint8)
        # rimozione punti di foreground isolati
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        # espansione del foreground
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

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
