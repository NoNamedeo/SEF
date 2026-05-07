from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from library.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from library.core.artifacts.Frame import Frame


class OpenCVInpaintingFrameProcessor(ISingleFrameProcessor):
    """
    Removes/inpaints regions of the frame specified by a mask.

    Mask convention:
        0   -> keep pixel
        255 -> inpaint pixel
    """

    def __init__(
        self,
        mask: np.ndarray,
        radius: float = 3.0, # radius of the context of the video around the mask that uses for inpainting
        method: int = 0, #INPAINT_TELEA (0) or INPAINT_NS (1) (Navier-Stroke based)
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        if mask is None:
            raise ValueError("Mask cannot be None.")

        if len(mask.shape) != 2:
            raise ValueError(
                "Mask must be a single-channel grayscale image."
            )

        self.mask = mask.astype(np.uint8)
        self.radius = radius
        match method:
            case 0: self.method = cv2.INPAINT_TELEA
            case 1: self.method = cv2.INPAINT_NS
            case _: self.method = cv2.INPAINT_TELEA

    def process(self, frame: Frame) -> Frame:
        image = frame.frame

        if image.shape[:2] != self.mask.shape[:2]:
            raise ValueError(
                "Image and mask dimensions do not match."
            )

        # Inpainting keeps the original frame geometry while replacing masked pixels.
        processed = cv2.inpaint(
            src=image,
            inpaintMask=self.mask,
            inpaintRadius=self.radius,
            flags=self.method,
        )

        return Frame(
            image=processed,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata={
                **dict(frame.metadata),
                "inpaint_radius": self.radius,
                "inpaint_method": (
                    "TELEA"
                    if self.method == cv2.INPAINT_TELEA
                    else "NS"
                ),
                "inpaint_mask_mean": float(np.mean(self.mask)),
            },
        )
