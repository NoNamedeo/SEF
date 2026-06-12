"""Mask artifacts and helpers used by frame-processing debug flows."""

from __future__ import annotations

from importlib import import_module

from sef.core._lazy_exports import install_lazy_exports

_EXPORTS = {
    "FrameMaskArtifact": ("sef.core.artifacts.mask.MaskArtifacts", "FrameMaskArtifact"),
    "IntermediateFrameArtifact": ("sef.core.artifacts.mask.MaskArtifacts", "IntermediateFrameArtifact"),
    "IntermediateFrameOverlay": ("sef.core.artifacts.mask.MaskArtifacts", "IntermediateFrameOverlay"),
    "MaskArray": ("sef.core.artifacts.mask.MaskOperations", "MaskArray"),
    "MaskArtifact": ("sef.core.artifacts.mask.MaskArtifacts", "MaskArtifact"),
    "MotionMaskArtifact": ("sef.core.artifacts.mask.MaskArtifacts", "MotionMaskArtifact"),
    "ProtectedRegionArtifact": ("sef.core.artifacts.mask.MaskArtifacts", "ProtectedRegionArtifact"),
    "ShapeSource": ("sef.core.artifacts.mask.MaskOperations", "ShapeSource"),
    "TargetMaskArtifact": ("sef.core.artifacts.mask.MaskArtifacts", "TargetMaskArtifact"),
    "ensure_shape_compatible": ("sef.core.artifacts.mask.MaskOperations", "ensure_shape_compatible"),
    "intersect_masks": ("sef.core.artifacts.mask.MaskOperations", "intersect_masks"),
    "merge_masks": ("sef.core.artifacts.mask.MaskOperations", "merge_masks"),
    "normalize_binary_mask": ("sef.core.artifacts.mask.MaskOperations", "normalize_binary_mask"),
    "spatial_shape_of": ("sef.core.artifacts.mask.MaskOperations", "spatial_shape_of"),
    "subtract_masks": ("sef.core.artifacts.mask.MaskOperations", "subtract_masks"),
    "validate_binary_mask": ("sef.core.artifacts.mask.MaskOperations", "validate_binary_mask"),
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


install_lazy_exports(__name__)
