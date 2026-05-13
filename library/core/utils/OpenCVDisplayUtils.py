from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class DisplayTransform:
    scale: float
    original_width: int
    original_height: int
    display_width: int
    display_height: int

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        *,
        max_width: int = 1280,
        max_height: int = 720,
    ) -> "DisplayTransform":
        original_height, original_width = image.shape[:2]
        scale = min(max_width / original_width, max_height / original_height, 1.0)
        display_width = max(int(round(original_width * scale)), 1)
        display_height = max(int(round(original_height * scale)), 1)
        return cls(
            scale=scale,
            original_width=original_width,
            original_height=original_height,
            display_width=display_width,
            display_height=display_height,
        )

    def resize_for_display(self, image: np.ndarray) -> np.ndarray:
        if self.scale == 1.0:
            return image
        return cv2.resize(
            image,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )

    def to_original_point(self, x: int, y: int) -> tuple[int, int]:
        original_x = int(round(x / self.scale))
        original_y = int(round(y / self.scale))
        return (
            min(max(original_x, 0), self.original_width - 1),
            min(max(original_y, 0), self.original_height - 1),
        )

    def to_original_length(self, length: int) -> int:
        return max(int(round(length / self.scale)), 1)

    def to_original_roi(self, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = roi
        original_x = min(max(int(round(x / self.scale)), 0), self.original_width - 1)
        original_y = min(max(int(round(y / self.scale)), 0), self.original_height - 1)
        original_w = min(max(int(round(w / self.scale)), 0), self.original_width - original_x)
        original_h = min(max(int(round(h / self.scale)), 0), self.original_height - original_y)
        return original_x, original_y, original_w, original_h

