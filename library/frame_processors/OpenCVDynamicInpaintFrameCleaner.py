from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame


class OpenCVDynamicInpaintFrameCleaner(IFrameCleaner):
    """
    Inpaints a dynamically tracked object.

    Workflow:
    - First frame: uses initial mask to locate object and initialize tracker
    - Next frames: updates tracker and rebuilds mask around new position
    """

    def __init__(
        self,
        mask: np.ndarray,
        radius: float = 3.0,
        method: int = 0,  # 0 = TELEA, 1 = NS
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)

        if mask is None:
            raise ValueError("Mask cannot be None.")

        if len(mask.shape) != 2:
            raise ValueError("Mask must be single-channel (grayscale).")

        self.initial_mask = mask.astype(np.uint8)
        self.radius = radius

        match method:
            case 0:
                self.method = cv2.INPAINT_TELEA
            case 1:
                self.method = cv2.INPAINT_NS
            case _:
                self.method = cv2.INPAINT_TELEA

        # tracking state
        self.tracker = None
        self.initialized = False
        self.bbox = None  # (x, y, w, h)

    def _init_tracker(self, frame_img: np.ndarray):
        # extract bbox from mask
        contours, _ = cv2.findContours(
            self.initial_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            raise ValueError("Mask does not contain any valid object region.")

        # take largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        self.bbox = (x, y, w, h)

        # create tracker (API compatibility handling)
        self.tracker = self._create_tracker()
        self.tracker.init(frame_img, self.bbox)

        self.initialized = True

    def _create_tracker(self):
        # OpenCV compatibility across versions
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()

        # fallback
        return cv2.TrackerKCF_create()

    def _bbox_to_mask(self, shape, bbox):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        x, y, w, h = map(int, bbox)
        mask[y:y+h, x:x+w] = 255
        return mask

    def clean(self, frame: Frame) -> Frame:
        image = frame.frame

        if image.shape[:2] != self.initial_mask.shape[:2]:
            raise ValueError("Image and initial mask dimensions do not match.")

        # init tracker on first frame
        if not self.initialized:
            self._init_tracker(image)

        # update tracker
        ok, bbox = self.tracker.update(image)

        if ok:
            self.bbox = bbox
        else:
            # fallback: keep previous bbox if tracking fails
            bbox = self.bbox

        dynamic_mask = self._bbox_to_mask(image.shape, bbox)

        cleaned = cv2.inpaint(
            src=image,
            inpaintMask=dynamic_mask,
            inpaintRadius=self.radius,
            flags=self.method,
        )

        return Frame(
            image=cleaned,
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
                "tracking_ok": bool(ok),
                "bbox": bbox,
                "mask_mean": float(np.mean(dynamic_mask)),
            },
        )