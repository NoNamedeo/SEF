from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from sef.builtin.frame_processors.dynamic_object_removal.config import DynamicObjectRemovalConfig


class ForegroundMaskExtractor(Protocol):
    """Extract candidate dynamic-object pixels from a frame and static background."""

    def extract(self, image: npt.NDArray, background: npt.NDArray) -> npt.NDArray[np.bool_]:
        """Return a 2D boolean foreground mask."""


class BackgroundDifferenceForegroundMaskExtractor:
    """Extract foreground pixels by thresholding absolute difference from background."""

    def __init__(self, config: DynamicObjectRemovalConfig) -> None:
        self._config = config

    def extract(self, image: npt.NDArray, background: npt.NDArray) -> npt.NDArray[np.bool_]:
        if image.shape != background.shape:
            raise ValueError(f"Frame shape {image.shape} does not match background shape {background.shape}.")

        difference = self._absolute_difference(image, background)
        difference_map = self._to_single_channel(difference)
        return difference_map > self._config.difference_threshold

    @staticmethod
    def _absolute_difference(image: npt.NDArray, background: npt.NDArray) -> npt.NDArray:
        if image.dtype == background.dtype and image.dtype == np.uint8:
            return cv2.absdiff(image, background)
        return np.abs(image.astype(np.float32) - background.astype(np.float32))

    def _to_single_channel(self, difference: npt.NDArray) -> npt.NDArray:
        if difference.ndim == 2:
            return difference
        if difference.ndim != 3:
            raise ValueError(f"Frame difference must be 2D or 3D; got shape {difference.shape}.")

        channels = int(difference.shape[2])
        if channels == 1:
            return difference[:, :, 0]
        if self._config.difference_mode == "mean":
            return np.mean(difference, axis=2)
        if self._config.difference_mode == "max":
            return np.max(difference, axis=2)
        if channels == 3:
            return cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(difference, cv2.COLOR_BGRA2GRAY)
        return np.max(difference, axis=2)
