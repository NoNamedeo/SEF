from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from library.core.interfaces.IVisualizer import IVisualizer
from library.core.visualization.VisualArtifact import ImageArtifact
from library.core.visualization.VisualizationContext import VisualizationContext


class MatplotlibArtifactVisualizer(IVisualizer):
    """Base helper for Matplotlib visualizers that emit PNG artifacts."""

    def _create_figure(
        self,
        *,
        figure_size: tuple[float, float],
        figure_facecolor: str,
    ) -> tuple[Figure, Any]:
        fig = Figure(figsize=figure_size, facecolor=figure_facecolor)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        return fig, ax

    def _build_image_artifact(
        self,
        fig: Figure,
        *,
        title: str,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ImageArtifact:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        return ImageArtifact(
            kind="image",
            title=title,
            description=description,
            metadata=dict(metadata or {}),
            mime_type="image/png",
            data=buffer.getvalue(),
        )

    @staticmethod
    def _artifact_metadata(
        context: VisualizationContext | None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = dict(extra or {})
        if context is None:
            return metadata
        if context.pipeline_id is not None:
            metadata.setdefault("pipeline_id", context.pipeline_id)
        if context.analyzer_name is not None:
            metadata.setdefault("analyzer_name", context.analyzer_name)
        if context.visualizer_name is not None:
            metadata.setdefault("visualizer_name", context.visualizer_name)
        if context.result_index is not None:
            metadata.setdefault("result_index", context.result_index)
        if context.source_metadata:
            metadata.setdefault("source_metadata", dict(context.source_metadata))
        if context.execution_metadata:
            metadata.setdefault("execution_metadata", dict(context.execution_metadata))
        return metadata
