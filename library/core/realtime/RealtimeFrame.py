from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping

import cv2
import numpy as np

FrameColorSpace = Literal["BGR", "RGB", "GRAY"]


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeFrame:
    """
    Immutable handoff object for realtime image previews.

    The frame carries pixels plus enough metadata for UI adapters to render it
    without depending on pipeline-specific data classes.
    """

    image: np.ndarray
    color_space: FrameColorSpace = "BGR"
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    produced_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise TypeError("RealtimeFrame.image must be a numpy.ndarray.")
        if self.image.ndim not in (2, 3):
            raise ValueError("RealtimeFrame.image must be a 2D or 3D image array.")
        if self.image.size <= 0:
            raise ValueError("RealtimeFrame.image cannot be empty.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_rgb(self) -> np.ndarray:
        """Return an RGB copy suitable for browser-oriented renderers."""
        if self.color_space == "RGB":
            return self.image.copy()
        if self.color_space == "BGR":
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        if self.color_space == "GRAY":
            return cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        raise ValueError(f"Unsupported frame color space: {self.color_space}")
