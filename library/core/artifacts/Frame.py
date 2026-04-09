from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Frame:
    """Video frame enriched with timeline metadata."""

    image: np.ndarray
    index: int | None = None
    timestamp_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame(self) -> np.ndarray:
        """Backward-compatible alias used by existing OpenCV code."""
        return self.image
