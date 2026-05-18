from __future__ import annotations

from library.core.pipeline.FrameExporterExecutor import FrameExporterExecutor
from library.core.pipeline.FramePipelineExecutor import FramePipelineExecutor
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from library.core.pipeline.PipelineComponentCapabilities import PipelineComponentCapabilities
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineExecutionResult import PipelineExecutionResult
from library.core.pipeline.SignalPipelineExecutor import SignalPipelineExecutor
from library.core.pipeline.StreamingSignalTailExecutor import StreamingSignalTailExecutor
from library.core.pipeline.VisualizationExecutor import VisualizationExecutor


class AdaptivePipelineExecutor:
    """
    Chooses the correct execution strategy for one pipeline run.

    Frame stages may stream until a batch-only boundary appears. The signal tail
    runs concurrently only when every downstream component supports streaming;
    otherwise the frame stream is materialized and the classic batch tail runs.
    """

    def __init__(
        self,
        *,
        context: PipelineContext,
        frame_pipeline_executor: FramePipelineExecutor,
        frame_exporter_executor: FrameExporterExecutor,
        signal_pipeline_executor: SignalPipelineExecutor,
        streaming_tail_executor: StreamingSignalTailExecutor,
        visualization_executor: VisualizationExecutor,
    ) -> None:
        self._context = context
        self._frame_pipeline_executor = frame_pipeline_executor
        self._frame_exporter_executor = frame_exporter_executor
        self._signal_pipeline_executor = signal_pipeline_executor
        self._streaming_tail_executor = streaming_tail_executor
        self._visualization_executor = visualization_executor

    def run(self) -> PipelineExecutionResult:
        """Execute the pipeline through the best available batch/streaming path."""
        intermediate_store = IntermediateFrameArtifactStore(self._context.intermediate_frame_capture)
        latency_policy = self._context.stream_runtime.latency_policy.create()
        frame_pipeline = self._frame_pipeline_executor.build(
            intermediate_store=intermediate_store,
            latency_policy=latency_policy,
        )

        if self._can_run_streaming_tail():
            return self._streaming_tail_executor.run(
                frame_pipeline=frame_pipeline,
                intermediate_store=intermediate_store,
                latency_policy=latency_policy,
            )

        materialized_buffer = self._frame_pipeline_executor.materialize(
            frame_pipeline.frame_buffer,
            frame_pipeline.pending_tasks,
            frame_buffers=frame_pipeline.frame_buffers,
            stage_name="frame_processing.materialize_final",
        )
        exported_buffer, frame_artifacts = self._frame_exporter_executor.run_batch(materialized_buffer)
        results = self._signal_pipeline_executor.run_batch(exported_buffer)
        intermediate_frames = intermediate_store.to_collection()
        final_artifacts = [*frame_artifacts, *self._visualization_executor.run_final_visualizers(results)]
        debug_artifacts = self._visualization_executor.run_intermediate_visualizers(intermediate_frames)
        return PipelineExecutionResult(
            results=tuple(results),
            final_artifacts=tuple(final_artifacts),
            debug_artifacts=tuple(debug_artifacts),
            intermediate_frames=intermediate_frames,
            latency_policy_metrics=latency_policy.metrics(),
        )

    def _can_run_streaming_tail(self) -> bool:
        return (
            PipelineComponentCapabilities.can_stream_signal_tail(self._context)
            and PipelineComponentCapabilities.can_stream_frame_exporters(self._context)
        )
