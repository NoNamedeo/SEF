"""Mask artifacts and helpers used by frame-processing debug flows."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FrameMaskArtifact": ("library.core.artifacts.mask.MaskArtifacts", "FrameMaskArtifact"),
    "IntermediateFrameArtifact": ("library.core.artifacts.mask.MaskArtifacts", "IntermediateFrameArtifact"),
    "IntermediateFrameOverlay": ("library.core.artifacts.mask.MaskArtifacts", "IntermediateFrameOverlay"),
    "MaskArray": ("library.core.artifacts.mask.MaskOperations", "MaskArray"),
    "MaskArtifact": ("library.core.artifacts.mask.MaskArtifacts", "MaskArtifact"),
    "MotionMaskArtifact": ("library.core.artifacts.mask.MaskArtifacts", "MotionMaskArtifact"),
    "ProtectedRegionArtifact": ("library.core.artifacts.mask.MaskArtifacts", "ProtectedRegionArtifact"),
    "ShapeSource": ("library.core.artifacts.mask.MaskOperations", "ShapeSource"),
    "TargetMaskArtifact": ("library.core.artifacts.mask.MaskArtifacts", "TargetMaskArtifact"),
    "ensure_shape_compatible": ("library.core.artifacts.mask.MaskOperations", "ensure_shape_compatible"),
    "intersect_masks": ("library.core.artifacts.mask.MaskOperations", "intersect_masks"),
    "merge_masks": ("library.core.artifacts.mask.MaskOperations", "merge_masks"),
    "normalize_binary_mask": ("library.core.artifacts.mask.MaskOperations", "normalize_binary_mask"),
    "spatial_shape_of": ("library.core.artifacts.mask.MaskOperations", "spatial_shape_of"),
    "subtract_masks": ("library.core.artifacts.mask.MaskOperations", "subtract_masks"),
    "validate_binary_mask": ("library.core.artifacts.mask.MaskOperations", "validate_binary_mask"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
