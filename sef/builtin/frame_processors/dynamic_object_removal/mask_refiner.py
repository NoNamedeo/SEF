from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from sef.builtin.frame_processors.dynamic_object_removal.config import DynamicObjectRemovalConfig
from sef.core.artifacts.mask.MaskOperations import normalize_binary_mask


@dataclass(frozen=True, slots=True)
class MaskRefinementResult:
    """Refined mask plus component-level diagnostics."""

    mask: npt.NDArray[np.bool_]
    component_count: int
    removed_component_count: int
    average_component_area: float


class MaskRefiner(Protocol):
    """Clean a raw binary foreground mask before reconstruction."""

    def refine(self, mask: npt.NDArray) -> MaskRefinementResult:
        """Return the refined mask and diagnostics."""


class MorphologicalMaskRefiner:
    """Refine foreground masks with morphology and small-component filtering."""

    def __init__(self, config: DynamicObjectRemovalConfig) -> None:
        self._config = config
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morph_kernel_size, config.morph_kernel_size),
        )

    def refine(self, mask: npt.NDArray) -> MaskRefinementResult:
        refined = normalize_binary_mask(mask, name="dynamic_mask").astype(np.uint8) * 255

        if self._config.opening_iterations > 0:
            refined = cv2.morphologyEx(
                refined,
                cv2.MORPH_OPEN,
                self._kernel,
                iterations=self._config.opening_iterations,
            )
        if self._config.closing_iterations > 0:
            refined = cv2.morphologyEx(
                refined,
                cv2.MORPH_CLOSE,
                self._kernel,
                iterations=self._config.closing_iterations,
            )
        if self._config.dilation_iterations > 0:
            refined = cv2.dilate(
                refined,
                self._kernel,
                iterations=self._config.dilation_iterations,
            )

        return self._remove_small_components(refined > 0)

    def _remove_small_components(self, mask: npt.NDArray[np.bool_]) -> MaskRefinementResult:
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if component_count <= 1:
            return MaskRefinementResult(
                mask=np.zeros(mask.shape, dtype=np.bool_),
                component_count=0,
                removed_component_count=0,
                average_component_area=0.0,
            )

        output = np.zeros(mask.shape, dtype=np.bool_)
        kept_areas: list[int] = []
        removed_component_count = 0
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < self._config.min_component_area:
                removed_component_count += 1
                continue
            output[labels == component_id] = True
            kept_areas.append(area)

        return MaskRefinementResult(
            mask=output,
            component_count=len(kept_areas),
            removed_component_count=removed_component_count,
            average_component_area=float(np.mean(kept_areas)) if kept_areas else 0.0,
        )
