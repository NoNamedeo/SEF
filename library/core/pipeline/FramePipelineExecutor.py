from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.pipeline.FrameProcessingStage import FrameProcessorExecutionContext
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from library.core.pipeline.LatencyPolicy import FrameLatencyPolicy
from library.core.pipeline.PipelineBuffers import PipelineBuffers
from library.core.pipeline.PipelineComponentCapabilities import PipelineComponentCapabilities
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineExecutionResult import FramePipelineResult, ThreadedStageTask
from library.core.pipeline.PipelineStageExecutor import PipelineStageExecutor


class FramePipelineExecutor:
    """
    Executes frame extraction and frame processors.

    The executor supports mixed pipelines: streaming stages are chained through
    bounded buffers, and batch-only stages trigger a materialization boundary.

    Ownership
    ---------
    This class owns only the frame segment of the pipeline:
    - frame extraction;
    - single-frame or frame-buffer processing;
    - conversion from streaming buffers to materialized buffers when required;
    - intermediate frame capture integration.

    It deliberately does not run exporters, signal extraction, analyzers, or
    visualizers. Those concerns are handled by dedicated collaborators.
    """

    def __init__(
        self,
        *,
        context: PipelineContext,
        stage_executor: PipelineStageExecutor,
    ) -> None:
        self._context = context
        self._stage_executor = stage_executor

    def build(
        self,
        *,
        intermediate_store: IntermediateFrameArtifactStore,
        latency_policy: FrameLatencyPolicy,
    ) -> FramePipelineResult:
        """
        Build the frame pipeline and return its current output buffer.

        Streaming processors are appended as pending threaded tasks. Batch-only
        processors force materialization of all pending upstream streams before
        processing continues.

        Parameters
        ----------
        intermediate_store:
            Store used by frame processors to publish debug/intermediate
            artifacts.
        latency_policy:
            Backpressure policy used by streaming frame extractors.

        Returns
        -------
        FramePipelineResult
            Latest frame buffer, pending streaming tasks, and every frame buffer
            that may need aborting if a concurrent stage fails.
        """
        frame_buffers: list[FrameBuffer] = []
        current_buffer, pending_tasks = self._extract_frames(frame_buffers, latency_policy)

        for processor_index, processor in enumerate(self._context.frame_processors):
            if PipelineComponentCapabilities.can_stream_frame_processor(processor):
                current_buffer = self._append_streaming_processor(
                    current_buffer,
                    pending_tasks,
                    frame_buffers,
                    processor=processor,
                    processor_index=processor_index,
                    intermediate_store=intermediate_store,
                )
                continue

            current_buffer = self.materialize(
                current_buffer,
                pending_tasks,
                frame_buffers=frame_buffers,
                stage_name=f"frame_processing[{processor_index}].materialize_input",
            )
            pending_tasks.clear()
            current_buffer = self._run_batch_processor(
                current_buffer,
                processor,
                processor_index=processor_index,
                intermediate_store=intermediate_store,
            )
            frame_buffers.append(current_buffer)

        return FramePipelineResult(
            frame_buffer=current_buffer,
            pending_tasks=tuple(pending_tasks),
            frame_buffers=frame_buffers,
        )

    def materialize(
        self,
        source_buffer: FrameBuffer,
        pending_tasks: list[ThreadedStageTask] | tuple[ThreadedStageTask, ...],
        *,
        frame_buffers: list[FrameBuffer],
        stage_name: str,
    ) -> FrameBuffer:
        """
        Drain pending streaming tasks and return a replayable frame buffer.

        Materialization is the explicit boundary between streaming and batch
        execution. The method starts all pending producers and a consumer that
        copies the source stream into a new closed ``FrameBuffer``. If any
        producer or the materializer fails, all known frame buffers are aborted
        to unblock waiting threads.
        """
        tasks = list(pending_tasks)
        if not tasks:
            return self._stage_executor.run(stage_name, lambda: PipelineBuffers.copy_frame_buffer(source_buffer))

        with ThreadPoolExecutor(max_workers=len(tasks) + 1, thread_name_prefix="sef-frame-boundary") as executor:
            futures = [task(executor) for task in tasks]
            materialized_future = executor.submit(
                lambda: self._stage_executor.run(stage_name, lambda: PipelineBuffers.copy_frame_buffer(source_buffer))
            )
            futures.append(materialized_future)
            try:
                for future in futures:
                    future.result()
            except Exception:
                PipelineBuffers.abort_all(frame_buffers, [], [])
                raise
        return materialized_future.result()

    def _extract_frames(
        self,
        frame_buffers: list[FrameBuffer],
        latency_policy: FrameLatencyPolicy,
    ) -> tuple[FrameBuffer, list[ThreadedStageTask]]:
        if PipelineComponentCapabilities.can_stream_frame_extractor(self._context.frame_extractor):
            output_buffer = FrameBuffer(buffer_size=self._context.stream_runtime.frame_buffer_size)
            frame_buffers.append(output_buffer)
            return output_buffer, [self._frame_extraction_task(output_buffer, latency_policy)]

        output_buffer = self._stage_executor.run("frame_extraction", lambda: self._context.frame_extractor.extract())
        frame_buffers.append(output_buffer)
        return output_buffer, []

    def _append_streaming_processor(
        self,
        input_buffer: FrameBuffer,
        pending_tasks: list[ThreadedStageTask],
        frame_buffers: list[FrameBuffer],
        *,
        processor: Any,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> FrameBuffer:
        output_buffer = FrameBuffer(buffer_size=self._context.stream_runtime.frame_buffer_size)
        frame_buffers.append(output_buffer)
        pending_tasks.append(
            self._frame_processor_task(
                input_buffer,
                output_buffer,
                processor=processor,
                processor_index=processor_index,
                intermediate_store=intermediate_store,
            )
        )
        return output_buffer

    def _run_batch_processor(
        self,
        buffer: FrameBuffer,
        processor: Any,
        *,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> FrameBuffer:
        return self._stage_executor.run(
            f"frame_processing[{processor_index}]",
            lambda: self._process_frame_buffer(
                buffer,
                processor,
                processor_index=processor_index,
                intermediate_store=intermediate_store,
            ),
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

    def _frame_extraction_task(
        self,
        output_buffer: FrameBuffer,
        latency_policy: FrameLatencyPolicy,
    ) -> ThreadedStageTask:
        return lambda executor: executor.submit(
            lambda: self._stage_executor.run(
                "frame_extraction",
                lambda: self._context.frame_extractor.extract_into(output_buffer, latency_policy),
            )
        )

    def _frame_processor_task(
        self,
        input_buffer: FrameBuffer,
        output_buffer: FrameBuffer,
        *,
        processor: Any,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> ThreadedStageTask:
        def submit_processor(executor: ThreadPoolExecutor) -> Future:
            return executor.submit(
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

        return submit_processor
