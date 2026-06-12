from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.artifacts.Frame import Frame
from sef.core.interfaces.BufferContracts import IBuffer, IFrameBuffer
from sef.core.interfaces.IFrameExporter import FrameExportContext
from sef.core.pipeline.FrameProcessingStage import FrameProcessorExecutionContext
from sef.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from sef.core.pipeline.LatencyPolicy import FrameLatencyPolicy
from sef.core.pipeline.PipelineBoundaryMaterializer import PipelineBoundaryMaterializer
from sef.core.pipeline.PipelineComponentCapabilities import PipelineComponentCapabilities
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineExecutionLookahead import PipelineExecutionLookahead
from sef.core.pipeline.PipelineExecutionPolicy import (
    PipelineExecutionEstimates,
    PipelineExecutionPolicy,
    PipelineStagePolicyContext,
)
from sef.core.pipeline.PipelineExecutionResources import PipelineExecutionResources
from sef.core.pipeline.PipelineRuntimeState import FrameRuntimeState, ThreadedStageTask
from sef.core.pipeline.PipelineStageExecutor import PipelineStageExecutor


class FrameSegmentExecutor:
    """
    Executes frame extraction, frame processors and frame exporters.

    This component owns frame-stage mechanics only. It delegates mode decisions
    to ``PipelineExecutionPolicy`` and boundary conversion to
    ``PipelineBoundaryMaterializer``.
    """

    def __init__(
        self,
        *,
        context: PipelineContext,
        stage_executor: PipelineStageExecutor,
        execution_policy: PipelineExecutionPolicy,
        lookahead: PipelineExecutionLookahead,
        estimates: PipelineExecutionEstimates,
        resources: PipelineExecutionResources,
        boundary_materializer: PipelineBoundaryMaterializer,
        pipeline_id: str | None,
        execution_metadata: Mapping[str, Any],
    ) -> None:
        self._context = context
        self._stage_executor = stage_executor
        self._execution_policy = execution_policy
        self._lookahead = lookahead
        self._estimates = estimates
        self._resources = resources
        self._boundary_materializer = boundary_materializer
        self._pipeline_id = pipeline_id
        self._execution_metadata = dict(execution_metadata)

    def run(
        self,
        *,
        latency_policy: FrameLatencyPolicy,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> FrameRuntimeState:
        """Run the complete frame-side segment and return its final state."""
        frames = self._run_frame_extractor(latency_policy)
        frames = self._run_frame_processors(frames, intermediate_store)
        return self._run_frame_exporters(frames)

    def _run_frame_extractor(self, latency_policy: FrameLatencyPolicy) -> FrameRuntimeState:
        decision = self._execution_policy.decide_source(
            PipelineStagePolicyContext(
                stage_id="frame_extraction",
                stage_group="frame_extractor",
                stage_streamable=PipelineComponentCapabilities.can_stream_frame_extractor(
                    self._context.frame_extractor
                ),
                downstream_streamable=self._lookahead.frame_successor_streamable(processor_index=0),
                estimated_queue_bytes=self._estimates.frame_queue_bytes,
                estimated_materialized_bytes=self._estimates.materialized_frame_bytes,
            )
        )
        if not decision.streams:
            buffer = self._stage_executor.run(
                "frame_extraction",
                lambda: self._context.frame_extractor.extract(),
            )
            self._resources.frame_buffers.append(buffer)
            return FrameRuntimeState(buffer=buffer, buffers=self._resources.frame_buffers)

        output = FrameBuffer(buffer_size=self._context.stream_runtime.frame_buffer_size)
        self._resources.frame_buffers.append(output)
        task = self._frame_extraction_task(output, latency_policy)
        return FrameRuntimeState(
            buffer=output,
            pending_tasks=[task],
            buffers=self._resources.frame_buffers,
        )

    def _run_frame_processors(
        self,
        state: FrameRuntimeState,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> FrameRuntimeState:
        for processor_index, processor in enumerate(self._context.frame_processors):
            decision = self._execution_policy.decide_stage(
                PipelineStagePolicyContext(
                    stage_id=f"frame_processing[{processor_index}]",
                    stage_group="frame_processors",
                    stage_streamable=PipelineComponentCapabilities.can_stream_frame_processor(processor),
                    input_is_streaming=state.is_streaming,
                    downstream_streamable=self._lookahead.frame_successor_streamable(
                        processor_index=processor_index + 1
                    ),
                    estimated_queue_bytes=self._estimates.frame_queue_bytes,
                    estimated_materialized_bytes=self._estimates.materialized_frame_bytes,
                )
            )
            if decision.streams:
                state = self._append_streaming_frame_processor(
                    state,
                    processor=processor,
                    processor_index=processor_index,
                    intermediate_store=intermediate_store,
                )
                continue

            buffer = self._boundary_materializer.materialize_frames(
                state,
                f"frame_processing[{processor_index}].materialize_input",
            )
            processed = self._stage_executor.run(
                f"frame_processing[{processor_index}]",
                lambda p=processor, b=buffer, idx=processor_index: self._process_frame_buffer(
                    b,
                    p,
                    processor_index=idx,
                    intermediate_store=intermediate_store,
                ),
            )
            self._resources.frame_buffers.append(processed)
            state = FrameRuntimeState(buffer=processed, buffers=self._resources.frame_buffers)
        return state

    def _run_frame_exporters(self, state: FrameRuntimeState) -> FrameRuntimeState:
        for exporter_index, exporter in enumerate(self._context.frame_exporters):
            decision = self._execution_policy.decide_stage(
                PipelineStagePolicyContext(
                    stage_id=f"frame_export[{exporter_index}]",
                    stage_group="frame_exporters",
                    stage_streamable=PipelineComponentCapabilities.can_stream_frame_exporter(exporter),
                    input_is_streaming=state.is_streaming,
                    downstream_streamable=self._lookahead.frame_export_successor_streamable(
                        exporter_index=exporter_index + 1
                    ),
                    estimated_queue_bytes=self._estimates.frame_queue_bytes,
                    estimated_materialized_bytes=self._estimates.materialized_frame_bytes,
                )
            )
            if decision.streams:
                state = self._append_streaming_frame_exporter(
                    state,
                    exporter=exporter,
                    exporter_index=exporter_index,
                )
                continue

            buffer = self._boundary_materializer.materialize_frames(
                state,
                f"frame_export[{exporter_index}].materialize_input",
            )
            result = self._stage_executor.run(
                f"frame_export[{exporter_index}]",
                lambda e=exporter, b=buffer: e.export(b, self._frame_export_context(e)),
            )
            self._resources.add_final_artifacts(result.artifacts)
            self._resources.frame_buffers.append(result.buffer)
            state = FrameRuntimeState(buffer=result.buffer, buffers=self._resources.frame_buffers)
        return state

    def _append_streaming_frame_processor(
        self,
        state: FrameRuntimeState,
        *,
        processor: Any,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> FrameRuntimeState:
        output = FrameBuffer(buffer_size=self._context.stream_runtime.frame_buffer_size)
        self._resources.frame_buffers.append(output)
        task = self._frame_processor_task(
            state.buffer,
            output,
            processor=processor,
            processor_index=processor_index,
            intermediate_store=intermediate_store,
        )
        return FrameRuntimeState(
            buffer=output,
            pending_tasks=[*state.pending_tasks, task],
            buffers=self._resources.frame_buffers,
        )

    def _append_streaming_frame_exporter(
        self,
        state: FrameRuntimeState,
        *,
        exporter: Any,
        exporter_index: int,
    ) -> FrameRuntimeState:
        output = FrameBuffer(buffer_size=self._context.stream_runtime.frame_buffer_size)
        self._resources.frame_buffers.append(output)
        task = self._frame_exporter_task(
            state.buffer,
            output,
            exporter=exporter,
            exporter_index=exporter_index,
        )
        return FrameRuntimeState(
            buffer=output,
            pending_tasks=[*state.pending_tasks, task],
            buffers=self._resources.frame_buffers,
        )

    @staticmethod
    def _process_frame_buffer(
        buffer: FrameBuffer,
        processor: Any,
        *,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> FrameBuffer:
        context = FrameProcessorExecutionContext(
            processor_index=processor_index,
            processor_name=type(processor).__name__,
            stage_name=f"frame_processing[{processor_index}].{type(processor).__name__}",
            intermediate_store=intermediate_store,
        )
        process_with_context = getattr(processor, "process_with_context", None)
        if callable(process_with_context):
            return process_with_context(buffer, context)
        return processor.process(buffer)

    def _frame_export_context(self, exporter: Any) -> FrameExportContext:
        return FrameExportContext(
            pipeline_id=self._pipeline_id,
            exporter_name=type(exporter).__name__,
            execution_metadata=dict(self._execution_metadata),
        )

    def _frame_extraction_task(
        self,
        output: IFrameBuffer,
        latency_policy: FrameLatencyPolicy,
    ) -> ThreadedStageTask:
        return lambda executor: executor.submit(
            lambda: self._stage_executor.run(
                "frame_extraction",
                lambda: self._context.frame_extractor.extract_into(output, latency_policy),
            )
        )

    def _frame_processor_task(
        self,
        input_buffer: Iterable[Frame],
        output_buffer: IBuffer[Frame],
        *,
        processor: Any,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> ThreadedStageTask:
        return lambda executor: executor.submit(
            lambda: self._stage_executor.run(
                f"frame_processing[{processor_index}]",
                lambda: processor.process_into(
                    input_buffer,
                    output_buffer,
                    processor_index=processor_index,
                    intermediate_store=intermediate_store,
                ),
            )
        )

    def _frame_exporter_task(
        self,
        input_buffer: Iterable[Frame],
        output_buffer: IBuffer[Frame],
        *,
        exporter: Any,
        exporter_index: int,
    ) -> ThreadedStageTask:
        return lambda executor: executor.submit(
            lambda: self._resources.add_final_artifacts(
                self._stage_executor.run(
                    f"frame_export[{exporter_index}]",
                    lambda: exporter.export_into(
                        input_buffer,
                        output_buffer,
                        self._frame_export_context(exporter),
                    ),
                )
            )
        )
