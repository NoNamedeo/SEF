from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sef.core.artifacts.intermediate_frame.IntermediateFrameArtifacts import IntermediateFrameArtifactCollection
from sef.core.interfaces.IData import IData
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineStageExecutor import PipelineStageExecutor
from sef.core.pipeline.VisualizerBinding import VisualizerBinding
from sef.core.visualization.VisualArtifact import VisualArtifact
from sef.core.visualization.VisualizationContext import VisualizationContext


class VisualizationExecutor:
    """
    Renders analysis and debug artifacts.

    This class owns visualizer binding resolution and ``VisualizationContext``
    creation. Pipeline execution can therefore treat visualization as a port,
    not as a collection of index and metadata rules.

    Binding model
    -------------
    Visualizers can be attached in two ways:
    - unbound visualizers render every analyzer result;
    - ``VisualizerBinding`` instances render only selected result indexes.

    The same binding rules are used for batch rendering and streaming target
    expansion, so visualizer behavior does not depend on execution mode.
    """

    def __init__(
        self,
        *,
        context: PipelineContext,
        stage_executor: PipelineStageExecutor,
        pipeline_id: str | None,
        execution_metadata: Mapping[str, Any],
    ) -> None:
        self._context = context
        self._stage_executor = stage_executor
        self._pipeline_id = pipeline_id
        self._execution_metadata = dict(execution_metadata)

    def run_final_visualizers(
        self,
        results: list[IData],
        *,
        skip_targets: set[tuple[int, int]] | None = None,
    ) -> list[VisualArtifact]:
        """
        Render artifacts for final analyzer results.

        Artifacts are returned in binding order and then result-index order.
        Invalid binding indexes are reported through the shared stage executor,
        preserving the same error shape as every other pipeline stage.
        """
        artifacts: list[VisualArtifact] = []
        skipped = skip_targets or set()
        for binding_index, binding in enumerate(self._bindings()):
            target_indexes = self._target_indexes(binding_index, binding, len(results))
            for result_index in target_indexes:
                if (binding_index, result_index) in skipped:
                    continue
                artifacts.extend(self._render_result(binding_index, binding, result_index, results[result_index]))
        return artifacts

    def run_intermediate_visualizers(
        self,
        intermediate_frames: IntermediateFrameArtifactCollection,
    ) -> list[VisualArtifact]:
        """
        Render debug artifacts from captured intermediate frame snapshots.

        Intermediate visualizers are intentionally separate from normal result
        visualizers because they consume diagnostic frame collections rather
        than analyzer output data.
        """
        if intermediate_frames.is_empty:
            return []

        artifacts: list[VisualArtifact] = []
        for visualizer_index, visualizer in enumerate(self._context.intermediate_frame_visualizers):
            rendered = self._stage_executor.run(
                f"visualisation.intermediate_frames[{visualizer_index}]",
                lambda v=visualizer: v.render(
                    intermediate_frames,
                    self.intermediate_context(v, intermediate_frames),
                ),
            )
            artifacts.extend(rendered)
        return artifacts

    def streaming_targets(self, result_count: int) -> list[tuple[int, VisualizerBinding, int]]:
        """
        Return stream visualizer bindings expanded to concrete result indexes.

        The returned tuples contain the binding index, binding object and target
        result index. ``SegmentedPipelineExecutor`` uses this information to
        create one data-buffer subscription per visualizer/result pair.
        """
        targets: list[tuple[int, VisualizerBinding, int]] = []
        for binding_index, binding in enumerate(self._bindings()):
            for result_index in self._target_indexes(binding_index, binding, result_count):
                targets.append((binding_index, binding, result_index))
        return targets

    def targets(self, result_count: int) -> list[tuple[int, VisualizerBinding, int]]:
        """Return every visualizer binding expanded to concrete result indexes."""
        return self.streaming_targets(result_count)

    def context_for(self, binding: VisualizerBinding, result_index: int, data: Any) -> VisualizationContext:
        """Create the context passed to a visualizer for one result stream."""
        analyzer_name = None
        if result_index < len(self._context.analyzers):
            analyzer_name = type(self._context.analyzers[result_index]).__name__

        source_metadata = getattr(data, "metadata", {})
        if not isinstance(source_metadata, Mapping):
            source_metadata = {}

        return VisualizationContext(
            pipeline_id=self._pipeline_id,
            analyzer_name=analyzer_name,
            visualizer_name=type(binding.visualizer).__name__,
            result_index=result_index,
            source_metadata=source_metadata,
            execution_metadata=dict(self._execution_metadata),
        )

    def intermediate_context(
        self,
        visualizer: Any,
        intermediate_frames: IntermediateFrameArtifactCollection,
    ) -> VisualizationContext:
        """Create a context for debug visualizers that consume intermediate frames."""
        return VisualizationContext(
            pipeline_id=self._pipeline_id,
            visualizer_name=type(visualizer).__name__,
            source_metadata=dict(intermediate_frames.metadata),
            execution_metadata=dict(self._execution_metadata),
            render_hints={"source": "intermediate_frames"},
        )

    def _bindings(self) -> list[VisualizerBinding]:
        return [
            *(VisualizerBinding(visualizer) for visualizer in self._context.visualizers),
            *self._context.visualizer_bindings,
        ]

    def _target_indexes(
        self,
        binding_index: int,
        binding: VisualizerBinding,
        result_count: int,
    ) -> tuple[int, ...]:
        return self._stage_executor.run(
            f"visualisation[{binding_index}].targets",
            lambda: self.resolve_targets(binding, result_count),
        )

    def _render_result(
        self,
        binding_index: int,
        binding: VisualizerBinding,
        result_index: int,
        data: IData,
    ) -> tuple[VisualArtifact, ...]:
        return self._stage_executor.run(
            f"visualisation[{binding_index}][{result_index}]",
            lambda: binding.visualizer.render(data, self.context_for(binding, result_index, data)),
        )

    @staticmethod
    def resolve_targets(binding: VisualizerBinding, result_count: int) -> tuple[int, ...]:
        """Resolve optional binding indexes against the available analyzer results."""
        return binding.target_indexes(result_count)
