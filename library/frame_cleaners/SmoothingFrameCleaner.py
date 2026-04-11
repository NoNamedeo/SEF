from __future__ import annotations

from typing import Any


import numpy as np

from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame


class SmoothingFrameCleaner(IFrameCleaner):
    """
    Temporal smoothing between consecutive frames.
    Reduces flicker and improves tracker stability by blending frames over time.
    This cleaner is stateful.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        config: dict[str, Any] | None = None,
    ):
        """
        alpha: weight of current frame (0..1) (higher = less smoothing, more responsiveness)
        """
        super().__init__(config)

        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        self.alpha = alpha
        self._previous_frame: np.ndarray | None = None

    def clean(self, frame: Frame) -> Frame:
        image = frame.frame.astype(np.float32)

        if self._previous_frame is None:
            smoothed = image
        else:
            smoothed = (
                self.alpha * image
                + (1.0 - self.alpha) * self._previous_frame
            )

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