"""Intermediate frame artifacts and composition helpers."""

from __future__ import annotations

from importlib import import_module

from sef.core._lazy_exports import install_lazy_exports

_EXPORTS = {
    "FrameComparisonPanel": (
        "sef.core.artifacts.intermediate_frame.IntermediateFrameComposition",
        "FrameComparisonPanel",
    ),
    "IntermediateFrameArtifactCollection": (
        "sef.core.artifacts.intermediate_frame.IntermediateFrameArtifacts",
        "IntermediateFrameArtifactCollection",
    ),
    "apply_mask_overlay": (
        "sef.core.artifacts.intermediate_frame.IntermediateFrameComposition",
        "apply_mask_overlay",
    ),
    "blend_overlay": (
        "sef.core.artifacts.intermediate_frame.IntermediateFrameComposition",
        "blend_overlay",
    ),
    "compose_image_grid": (
        "sef.core.artifacts.intermediate_frame.IntermediateFrameComposition",
        "compose_image_grid",
    ),
    "compose_intermediate_frame_comparison": (
        "sef.core.artifacts.intermediate_frame.IntermediateFrameComposition",
        "compose_intermediate_frame_comparison",
    ),
    "compose_side_by_side": (
        "sef.core.artifacts.intermediate_frame.IntermediateFrameComposition",
        "compose_side_by_side",
    ),
    "encode_png": (
        "sef.core.artifacts.intermediate_frame.IntermediateFrameComposition",
        "encode_png",
    ),
    "to_display_bgr": (
        "sef.core.artifacts.intermediate_frame.IntermediateFrameComposition",
        "to_display_bgr",
    ),
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
