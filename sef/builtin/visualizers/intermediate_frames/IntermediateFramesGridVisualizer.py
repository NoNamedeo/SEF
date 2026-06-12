from __future__ import annotations

from typing import Any

from sef.core.artifacts.intermediate_frame.IntermediateFrameComposition import (
    compose_image_grid,
    compose_intermediate_frame_comparison,
    encode_png,
)
from sef.core.interfaces.IData import IData
from sef.core.visualization.VisualArtifact import ArtifactRole, ImageArtifact, VisualArtifact
from sef.core.visualization.VisualizationContext import VisualizationContext
from sef.builtin.visualizers.intermediate_frames.IntermediateFramesVisualizer import (
    IntermediateFramesVisualizer,
    _artifact_grid_label,
)


class IntermediateFramesGridVisualizer(IntermediateFramesVisualizer):
    """Render captured intermediate frame comparisons as one bounded PNG grid."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.columns = self._positive_int(self.config.get("columns", 2), field_name="columns")
        self.max_cell_width = self._optional_positive_int(
            self.config.get("max_cell_width", 720),
            field_name="max_cell_width",
        )
        self.gap = self._non_negative_int(self.config.get("gap", 10), field_name="gap")
        self.show_cell_labels = bool(self.config.get("show_cell_labels", False))
        if "max_artifacts" not in self.config:
            self.max_artifacts = 24

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        artifacts = self._selected_artifacts(data)
        if not artifacts:
            return ()

        comparisons = [
            compose_intermediate_frame_comparison(
                artifact,
                show_labels=self.show_labels,
                include_masks=self.include_masks,
                include_overlays=self.include_overlays,
                max_panel_width=self.max_panel_width,
            )
            for artifact in artifacts
        ]
        labels = tuple(_artifact_grid_label(artifact) for artifact in artifacts) if self.show_cell_labels else None
        grid = compose_image_grid(
            comparisons,
            labels=labels,
            columns=self.columns,
            max_cell_width=self.max_cell_width,
            gap=self.gap,
        )
        return (
            ImageArtifact(
                kind="image",
                role=ArtifactRole.DEBUG,
                title="Intermediate frame grid",
                description="Sampled preprocessing stages rendered as a comparison grid.",
                metadata=self._artifact_metadata(
                    context,
                    {
                        "data_type": type(data).__name__,
                        "artifact_count": len(artifacts),
                        "stage_names": tuple(dict.fromkeys(artifact.stage_name for artifact in artifacts)),
                        "frame_indices": tuple(dict.fromkeys(artifact.frame_index for artifact in artifacts)),
                        "columns": self.columns,
                    },
                ),
                mime_type="image/png",
                data=encode_png(grid),
            ),
        )

    @staticmethod
    def _positive_int(value: Any, *, field_name: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        return parsed

    @staticmethod
    def _non_negative_int(value: Any, *, field_name: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{field_name} cannot be negative.")
        return parsed
