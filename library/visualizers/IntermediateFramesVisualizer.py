from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from library.core.artifacts.IntermediateFrameArtifacts import IntermediateFrameArtifactCollection
from library.core.artifacts.IntermediateFrameComposition import (
    compose_intermediate_frame_comparison,
    encode_png,
)
from library.core.artifacts.MaskArtifacts import IntermediateFrameArtifact
from library.core.interfaces.IData import IData
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.visualization.VisualArtifact import ImageArtifact, VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext


class IntermediateFramesVisualizer(IVisualizer):
    """Render each intermediate frame artifact as a standalone comparison PNG."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.show_labels = bool(self.config.get("show_labels", True))
        self.include_masks = bool(self.config.get("include_masks", True))
        self.include_overlays = bool(self.config.get("include_overlays", True))
        self.max_panel_width = self._optional_positive_int(
            self.config.get("max_panel_width", 480),
            field_name="max_panel_width",
        )
        self.max_artifacts = self._optional_non_negative_int(
            self.config.get("max_artifacts"),
            field_name="max_artifacts",
        )

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        artifacts = self._selected_artifacts(data)
        rendered: list[VisualArtifact] = []
        for artifact in artifacts:
            comparison = compose_intermediate_frame_comparison(
                artifact,
                show_labels=self.show_labels,
                include_masks=self.include_masks,
                include_overlays=self.include_overlays,
                max_panel_width=self.max_panel_width,
            )
            rendered.append(
                ImageArtifact(
                    kind="image",
                    title=self._title(artifact),
                    description="Intermediate frame comparison generated during frame processing.",
                    metadata=self._artifact_metadata(
                        context,
                        {
                            "data_type": type(data).__name__,
                            "stage_name": artifact.stage_name,
                            "frame_index": artifact.frame_index,
                            "timestamp_seconds": artifact.timestamp_seconds,
                            "mask_count": len(artifact.masks),
                            "overlay_count": len(artifact.overlays),
                            "stage_metadata": dict(artifact.stage_metadata),
                        },
                    ),
                    mime_type="image/png",
                    data=encode_png(comparison),
                )
            )
        return tuple(rendered)

    def _selected_artifacts(self, data: IData) -> tuple[IntermediateFrameArtifact, ...]:
        artifacts = self._artifacts_from(data)
        if self.max_artifacts is None:
            return artifacts
        return artifacts[: self.max_artifacts]

    @staticmethod
    def _artifacts_from(data: IData) -> tuple[IntermediateFrameArtifact, ...]:
        if isinstance(data, IntermediateFrameArtifact):
            return (data,)
        if isinstance(data, IntermediateFrameArtifactCollection):
            return data.artifacts
        raise TypeError(
            "IntermediateFramesVisualizer requires IntermediateFrameArtifact or "
            f"IntermediateFrameArtifactCollection, got {type(data).__name__}."
        )

    @staticmethod
    def _title(artifact: IntermediateFrameArtifact) -> str:
        frame_label = "unknown frame" if artifact.frame_index is None else f"frame {artifact.frame_index}"
        return f"{artifact.stage_name} ({frame_label})"

    @staticmethod
    def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
        if value is None:
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        return parsed

    @staticmethod
    def _optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
        if value is None:
            return None
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{field_name} cannot be negative.")
        return parsed

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
        if context.render_hints:
            metadata.setdefault("render_hints", dict(context.render_hints))
        return metadata


def _artifact_grid_label(artifact: IntermediateFrameArtifact) -> str:
    frame_label = "?" if artifact.frame_index is None else str(artifact.frame_index)
    return f"{frame_label} | {artifact.stage_name}"
