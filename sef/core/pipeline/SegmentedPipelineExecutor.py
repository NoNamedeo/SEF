from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from library.core.pipeline.AnalysisSegmentExecutor import AnalysisSegmentExecutor
from library.core.pipeline.FrameSegmentExecutor import FrameSegmentExecutor
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from library.core.pipeline.PipelineBoundaryMaterializer import PipelineBoundaryMaterializer
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineExecutionLookahead import PipelineExecutionLookahead
from library.core.pipeline.PipelineExecutionPolicy import (
    PipelineExecutionEstimates,
    PipelineExecutionPolicy,
)
from library.core.pipeline.PipelineExecutionResources import PipelineExecutionResources
from library.core.pipeline.PipelineExecutionResult import PipelineExecutionResult
from library.core.pipeline.PipelineStageExecutor import PipelineStageExecutor
from library.core.pipeline.SignalSegmentExecutor import SignalSegmentExecutor
from library.core.pipeline.VisualizationExecutor import VisualizationExecutor


class SegmentedPipelineExecutor:
    """
    Composes the segment executors that run one pipeline.

    The class deliberately owns only high-level sequencing. Frame, signal,
    analysis, materialization, artifact collection and execution-mode policy
    live in dedicated collaborators.
    """

    def __init__(
        self,
        *,
        context: PipelineContext,
        stage_executor: PipelineStageExecutor,
        visualization_executor: VisualizationExecutor,
        execution_policy: PipelineExecutionPolicy,
        pipeline_id: str | None,
        execution_metadata: Mapping[str, Any],
    ) -> None:
        self._context = context
        self._stage_executor = stage_executor
        self._visualization_executor = visualization_executor
        self._execution_policy = execution_policy
        self._pipeline_id = pipeline_id
        self._execution_metadata = dict(execution_metadata)

    def run(self) -> PipelineExecutionResult:
        """Execute all pipeline segments and assemble the raw runtime result."""
        resources = PipelineExecutionResources()
        estimates = PipelineExecutionEstimates.from_context(self._context)
        lookahead = PipelineExecutionLookahead(self._context)
        boundary_materializer = PipelineBoundaryMaterializer(
            stage_executor=self._stage_executor,
            resources=resources,
            signal_buffer_size=self._context.stream_runtime.signal_buffer_size,
        )
        intermediate_store = IntermediateFrameArtifactStore(
            self._context.intermediate_frame_capture
        )
        latency_policy = self._context.stream_runtime.latency_policy.create()

        frames = self._frame_executor(
            resources=resources,
            estimates=estimates,
            lookahead=lookahead,
            boundary_materializer=boundary_materializer,
        ).run(latency_policy=latency_policy, intermediate_store=intermediate_store)
        signal = self._signal_executor(
            resources=resources,
            estimates=estimates,
            lookahead=lookahead,
            boundary_materializer=boundary_materializer,
        ).run(frames)
        results = self._analysis_executor(
            resources=resources,
            estimates=estimates,
            boundary_materializer=boundary_materializer,
        ).run(signal)

        intermediate_frames = intermediate_store.to_collection()
        debug_artifacts = self._visualization_executor.run_intermediate_visualizers(
            intermediate_frames
        )
        return PipelineExecutionResult(
            results=tuple(results),
            final_artifacts=resources.final_artifacts,
            debug_artifacts=tuple(debug_artifacts),
            intermediate_frames=intermediate_frames,
            latency_policy_metrics=latency_policy.metrics(),
        )

    def _frame_executor(
        self,
        *,
        resources: PipelineExecutionResources,
        estimates: PipelineExecutionEstimates,
        lookahead: PipelineExecutionLookahead,
        boundary_materializer: PipelineBoundaryMaterializer,
    ) -> FrameSegmentExecutor:
        return FrameSegmentExecutor(
            context=self._context,
            stage_executor=self._stage_executor,
            execution_policy=self._execution_policy,
            lookahead=lookahead,
            estimates=estimates,
            resources=resources,
            boundary_materializer=boundary_materializer,
            pipeline_id=self._pipeline_id,
            execution_metadata=self._execution_metadata,
        )

    def _signal_executor(
        self,
        *,
        resources: PipelineExecutionResources,
        estimates: PipelineExecutionEstimates,
        lookahead: PipelineExecutionLookahead,
        boundary_materializer: PipelineBoundaryMaterializer,
    ) -> SignalSegmentExecutor:
        return SignalSegmentExecutor(
            context=self._context,
            stage_executor=self._stage_executor,
            execution_policy=self._execution_policy,
            lookahead=lookahead,
            estimates=estimates,
            resources=resources,
            boundary_materializer=boundary_materializer,
        )

    def _analysis_executor(
        self,
        *,
        resources: PipelineExecutionResources,
        estimates: PipelineExecutionEstimates,
        boundary_materializer: PipelineBoundaryMaterializer,
    ) -> AnalysisSegmentExecutor:
        return AnalysisSegmentExecutor(
            context=self._context,
            stage_executor=self._stage_executor,
            visualization_executor=self._visualization_executor,
            execution_policy=self._execution_policy,
            estimates=estimates,
            resources=resources,
            boundary_materializer=boundary_materializer,
        )
