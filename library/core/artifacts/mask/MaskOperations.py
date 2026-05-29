from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

MaskArray = npt.NDArray[np.bool_]
ShapeSource = npt.NDArray[Any] | Sequence[int]

_ALLOWED_BINARY_VALUES = (0, 1, 255)


def validate_binary_mask(mask: npt.NDArray[Any], *, name: str = "mask") -> None:
    """
    Validate that an array is a non-empty 2D binary mask.

    Boolean, 0/1, and OpenCV-style 0/255 masks are accepted. The function does
    not normalize or copy the data; use `normalize_binary_mask` when consumers
    need a canonical boolean representation.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D binary mask; got shape {mask.shape}.")
    if mask.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if np.issubdtype(mask.dtype, np.bool_):
        return
    if not np.issubdtype(mask.dtype, np.number):
        raise TypeError(f"{name} must contain boolean or numeric binary values; got dtype {mask.dtype}.")
    if np.issubdtype(mask.dtype, np.floating) and not np.isfinite(mask).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if not np.isin(mask, _ALLOWED_BINARY_VALUES).all():
        raise ValueError(f"{name} must contain only binary values: {_ALLOWED_BINARY_VALUES}.")


def normalize_binary_mask(mask: npt.NDArray[Any], *, name: str = "mask", readonly: bool = False) -> MaskArray:
    """
    Return a canonical boolean copy of a valid binary mask.

    A defensive copy prevents artifact instances from changing when callers
    mutate the original array after construction.
    """
    validate_binary_mask(mask, name=name)
    normalized = np.array(mask, dtype=np.bool_, copy=True)
    if readonly:
        normalized.setflags(write=False)
    return normalized


def spatial_shape_of(value: ShapeSource, *, name: str = "shape") -> tuple[int, int]:
    """
    Return the `(height, width)` portion of an array or shape tuple.

    This supports mask-to-mask and mask-to-frame compatibility checks without
    requiring frame arrays to be reduced to 2D first.
    """
    shape = value.shape if isinstance(value, np.ndarray) else tuple(int(dimension) for dimension in value)
    if len(shape) < 2:
        raise ValueError(f"{name} must expose at least two spatial dimensions; got {shape}.")

    height = int(shape[0])
    width = int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must have positive spatial dimensions; got {(height, width)}.")
    return height, width


def ensure_shape_compatible(
    reference: ShapeSource,
    candidate: ShapeSource,
    *,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> None:
    """Raise a clear error when two masks or frame-like arrays do not share spatial shape."""
    reference_shape = spatial_shape_of(reference, name=reference_name)
    candidate_shape = spatial_shape_of(candidate, name=candidate_name)
    if reference_shape != candidate_shape:
        raise ValueError(
            f"{candidate_name} spatial shape {candidate_shape} does not match "
            f"{reference_name} spatial shape {reference_shape}."
        )


def merge_masks(*masks: npt.NDArray[Any]) -> MaskArray:
    """Return the union of one or more compatible binary masks."""
    normalized_masks = _normalize_compatible_masks(*masks)
    merged = np.zeros(normalized_masks[0].shape, dtype=np.bool_)
    for mask in normalized_masks:
        np.logical_or(merged, mask, out=merged)
    return merged


def intersect_masks(*masks: npt.NDArray[Any]) -> MaskArray:
    """Return the intersection of one or more compatible binary masks."""
    normalized_masks = _normalize_compatible_masks(*masks)
    intersection = np.ones(normalized_masks[0].shape, dtype=np.bool_)
    for mask in normalized_masks:
        np.logical_and(intersection, mask, out=intersection)
    return intersection


def subtract_masks(base_mask: npt.NDArray[Any], *excluded_masks: npt.NDArray[Any]) -> MaskArray:
    """Return `base_mask` with every active pixel in `excluded_masks` removed."""
    normalized_base = normalize_binary_mask(base_mask, name="base_mask")
    result = normalized_base.copy()
    for index, excluded_mask in enumerate(excluded_masks):
        normalized_excluded = normalize_binary_mask(excluded_mask, name=f"excluded_masks[{index}]")
        ensure_shape_compatible(normalized_base, normalized_excluded, reference_name="base_mask", candidate_name=f"excluded_masks[{index}]")
        np.logical_and(result, np.logical_not(normalized_excluded), out=result)
    return result


def _normalize_compatible_masks(*masks: npt.NDArray[Any]) -> tuple[MaskArray, ...]:
    if not masks:
        raise ValueError("At least one mask is required.")

    normalized_masks = tuple(
        normalize_binary_mask(mask, name=f"masks[{index}]")
        for index, mask in enumerate(masks)
    )
    reference = normalized_masks[0]
    for index, mask in enumerate(normalized_masks[1:], start=1):
        ensure_shape_compatible(reference, mask, reference_name="masks[0]", candidate_name=f"masks[{index}]")
    return normalized_masks
