from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from sef.core.artifacts.mask.MaskOperations import ensure_shape_compatible, normalize_binary_mask


class RegionReconstructor(Protocol):
    """Reconstruct masked frame regions using an estimated static background."""

    def reconstruct(
        self,
        image: npt.NDArray,
        background: npt.NDArray,
        mask: npt.NDArray,
    ) -> npt.NDArray:
        """Return an image with masked pixels reconstructed."""


class BackgroundReplacementRegionReconstructor:
    """Replace only active mask pixels with pixels from the estimated background."""

    def reconstruct(
        self,
        image: npt.NDArray,
        background: npt.NDArray,
        mask: npt.NDArray,
    ) -> npt.NDArray:
        if image.shape != background.shape:
            raise ValueError(f"Frame shape {image.shape} does not match background shape {background.shape}.")
        ensure_shape_compatible(image, mask, reference_name="image", candidate_name="mask")

        normalized_mask = normalize_binary_mask(mask, name="effective_removal_mask")
        if not np.any(normalized_mask):
            return image

        cleaned = image.copy()
        cleaned[normalized_mask] = background[normalized_mask]
        return cleaned
