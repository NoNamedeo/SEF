from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from sef.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from sef.core.artifacts.Frame import Frame


class OpenCVBackgroundReplacementFrameProcessor(ISingleFrameProcessor):
    """
    Replaces masked regions of the frame using a clean background image.

    Mask convention:
        0   -> keep original frame
        255 -> replace using clean background
    """

    def __init__(
        self,
        background_image_path: str,
        mask: np.ndarray,
        resize: tuple[int, int] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        if mask is None:
            raise ValueError("Mask cannot be None.")

        if len(mask.shape) != 2:
            raise ValueError(
                "Mask must be single-channel."
            )

        background = cv2.imread(background_image_path)

        if background is None:
            raise ValueError(
                f"Cannot load image: {background_image_path}"
            )

        if resize is not None:
            background = cv2.resize(background, resize)

        self.background = background
        self.mask = mask.astype(np.uint8)

    def process(self, frame: Frame) -> Frame:

        image = frame.frame

        if image.shape[:2] != self.mask.shape[:2]:
            raise ValueError(
                "Image and mask dimensions do not match."
            )

        if image.shape[:2] != self.background.shape[:2]:
            raise ValueError(
                "Image and background dimensions do not match."
            )

        cleaned = image.copy()

        # sostituisce solo i pixel mascherati
        cleaned[self.mask > 0] = self.background[self.mask > 0]

        return Frame(
            image=cleaned,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata={
                **dict(frame.metadata),
                "background_replacement": True,
                "mask_mean": float(np.mean(self.mask)),
            },
        )