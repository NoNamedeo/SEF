from __future__ import annotations

from typing import Any

import numpy as np

from sef.core.artifacts.Frame import Frame
from sef.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor


class SmoothingFrameProcessor(ISingleFrameProcessor):
    """
    Temporal smoothing between consecutive frames.
    Reduces flicker and improves tracker stability by blending frames over time.
    """

    def __init__(
        self,
        alpha: float = 0.9,
        reset_threshold: float = 40.0,
        config: dict[str, Any] | None = None,
    ):
        """
        alpha: weight of current frame (0 < alpha <= 1) (higher = less smoothing, more responsiveness)
        """
        super().__init__(config)

        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        self.alpha = alpha
        self.reset_threshold = reset_threshold
        self._previous_frame: np.ndarray | None = None

    def process(self, frame: Frame) -> Frame:
        image = frame.frame.astype(np.float32)

        # differenza tra frame attuale e passato (se eccessiva (tipo cambio video), resetta)
        difference = 0
        if self._previous_frame is not None:
            difference = np.mean(np.abs(image - self._previous_frame))

        if self._previous_frame is None or difference > self.reset_threshold:
            smoothed = image
        else:
            # mischia i due frame
            smoothed = self.alpha * image + (1.0 - self.alpha) * self._previous_frame

        self._previous_frame = smoothed

        smoothed_uint8 = np.clip(smoothed, 0, 255).astype(np.uint8)

        return Frame(
            image=smoothed_uint8,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata={
                **dict(frame.metadata),
                "smoothing_alpha": self.alpha,
            },
        )
