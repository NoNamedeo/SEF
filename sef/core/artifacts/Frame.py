from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Frame:
    """
    Video frame plus timeline and plugin metadata.

    The `image` array is intentionally not copied by the value object. Producers
    and consumers should document whether they mutate frame pixels in place or
    return new arrays.

    Attributes
    ----------
    image:
        Pixel array, typically an OpenCV-compatible NumPy image.
    index:
        Optional source-order index.
    timestamp_seconds:
        Optional source timestamp in seconds.
    metadata:
        Mutable metadata dictionary for plugin-specific annotations.
    """

    image: np.ndarray
    index: int | None = None
    timestamp_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame(self) -> np.ndarray:
        """Backward-compatible alias for `image` used by existing OpenCV code."""
        return self.image
