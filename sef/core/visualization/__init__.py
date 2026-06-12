"""UI-agnostic visualization contracts and pipeline output values.

Visualizers should return `VisualArtifact` instances instead of manipulating a
specific UI toolkit. This lets Streamlit, notebooks, web APIs, CLIs, and file
exporters render the same completed pipeline output without coupling the core
runtime to presentation infrastructure.
"""

from __future__ import annotations

from importlib import import_module

from sef.core._lazy_exports import install_lazy_exports

_EXPORTS = {
    "ArtifactRole": ("sef.core.visualization.VisualArtifact", "ArtifactRole"),
    "DeferredVideoArtifact": (
        "sef.core.visualization.VisualArtifact",
        "DeferredVideoArtifact",
    ),
    "ImageArtifact": ("sef.core.visualization.VisualArtifact", "ImageArtifact"),
    "JsonArtifact": ("sef.core.visualization.VisualArtifact", "JsonArtifact"),
    "PipelineOutputs": ("sef.core.visualization.PipelineOutputs", "PipelineOutputs"),
    "PipelineRunMetadata": (
        "sef.core.visualization.PipelineRunMetadata",
        "PipelineRunMetadata",
    ),
    "TableArtifact": ("sef.core.visualization.VisualArtifact", "TableArtifact"),
    "TextArtifact": ("sef.core.visualization.VisualArtifact", "TextArtifact"),
    "VIDEO_ARTIFACT_TYPES": (
        "sef.core.visualization.VisualArtifact",
        "VIDEO_ARTIFACT_TYPES",
    ),
    "VideoArtifact": ("sef.core.visualization.VisualArtifact", "VideoArtifact"),
    "VideoFileArtifact": ("sef.core.visualization.VisualArtifact", "VideoFileArtifact"),
    "VideoLikeArtifact": ("sef.core.visualization.VisualArtifact", "VideoLikeArtifact"),
    "VisualArtifact": ("sef.core.visualization.VisualArtifact", "VisualArtifact"),
    "VisualizationContext": (
        "sef.core.visualization.VisualizationContext",
        "VisualizationContext",
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


