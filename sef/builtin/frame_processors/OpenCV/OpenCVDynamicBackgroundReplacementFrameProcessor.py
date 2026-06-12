from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from sef.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from sef.core.artifacts.Frame import Frame


class OpenCVDynamicBackgroundReplacementFrameProcessor(ISingleFrameProcessor):
    """
    Dynamically tracks an object starting from an initial mask
    and replaces it with a static background image.

    Mask convention:
        0   -> keep original frame
        255 -> replace with background
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
            raise ValueError("Mask must be single-channel.")

        background = cv2.imread(background_image_path)
        if background is None:
            raise ValueError(f"Cannot load image: {background_image_path}")

        if resize is not None:
            background = cv2.resize(background, resize)

        self.background = background
        self.initial_mask = mask.astype(np.uint8)

        # tracking state
        self.tracker = None
        self.initialized = False
        self.bbox = None

    def _create_tracker(self):
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        return cv2.TrackerKCF_create()

    def _init_tracker(self, frame: np.ndarray):
        contours, _ = cv2.findContours(
            self.initial_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            raise ValueError("Initial mask does not contain any object.")

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        self.bbox = (x, y, w, h)

        self.tracker = self._create_tracker()
        self.tracker.init(frame, self.bbox)

        self.initialized = True

    def _bbox_to_mask(self, shape, bbox):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        x, y, w, h = map(int, bbox)
        mask[y:y+h, x:x+w] = 255
        return mask

    def process(self, frame: Frame) -> Frame:
        image = frame.frame

        if image.shape[:2] != self.initial_mask.shape[:2]:
            raise ValueError("Image and mask dimensions do not match.")

        if image.shape[:2] != self.background.shape[:2]:
            raise ValueError("Image and background dimensions do not match.")

        # init tracker once
        if not self.initialized:
            self._init_tracker(image)

        ok, bbox = self.tracker.update(image)

        if ok:
            self.bbox = bbox

        dynamic_mask = self._bbox_to_mask(image.shape, self.bbox)

        cleaned = image.copy()

        # replace only tracked region
        cleaned[dynamic_mask > 0] = self.background[dynamic_mask > 0]

        return Frame(
            image=cleaned,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata={
                **dict(frame.metadata),
                "background_replacement_dynamic": True,
                "tracking_ok": bool(ok),
                "bbox": self.bbox,
                "mask_mean": float(np.mean(dynamic_mask)),
            },
        )