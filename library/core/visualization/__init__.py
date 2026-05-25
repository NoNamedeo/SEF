from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ArtifactRole": ("library.core.visualization.VisualArtifact", "ArtifactRole"),
    "DeferredVideoArtifact": (
        "library.core.visualization.VisualArtifact",
        "DeferredVideoArtifact",
    ),
    "ImageArtifact": ("library.core.visualization.VisualArtifact", "ImageArtifact"),
    "JsonArtifact": ("library.core.visualization.VisualArtifact", "JsonArtifact"),
    "PipelineOutputs": ("library.core.visualization.PipelineOutputs", "PipelineOutputs"),
    "PipelineRunMetadata": (
        "library.core.visualization.PipelineRunMetadata",
        "PipelineRunMetadata",
    ),
    "TableArtifact": ("library.core.visualization.VisualArtifact", "TableArtifact"),
    "TextArtifact": ("library.core.visualization.VisualArtifact", "TextArtifact"),
    "VIDEO_ARTIFACT_TYPES": (
        "library.core.visualization.VisualArtifact",
        "VIDEO_ARTIFACT_TYPES",
    ),
    "VideoArtifact": ("library.core.visualization.VisualArtifact", "VideoArtifact"),
    "VideoFileArtifact": ("library.core.visualization.VisualArtifact", "VideoFileArtifact"),
    "VideoLikeArtifact": ("library.core.visualization.VisualArtifact", "VideoLikeArtifact"),
    "VisualArtifact": ("library.core.visualization.VisualArtifact", "VisualArtifact"),
    "VisualizationContext": (
        "library.core.visualization.VisualizationContext",
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


for _name in __all__:
    __getattr__(_name)
del _name
