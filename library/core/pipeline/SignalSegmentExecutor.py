from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.ISignalSample import ISignalSample
from library.core.pipeline.PipelineBoundaryMaterializer import PipelineBoundaryMaterializer
from library.core.pipeline.PipelineComponentCapabilities import PipelineComponentCapabilities
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineExecutionLookahead import PipelineExecutionLookahead
from library.core.pipeline.PipelineExecutionPolicy import (
    PipelineExecutionEstimates,
    PipelineExecutionPolicy,
    PipelineStagePolicyContext,
)
from library.core.pipeline.PipelineExecutionResources import PipelineExecutionResources
from library.core.pipeline.PipelineRuntimeState import FrameRuntimeState, SignalRuntimeState, ThreadedStageTask
from library.core.pipeline.PipelineStageExecutor import PipelineStageExecutor


class SignalSegmentExecutor:
    """
    Executes signal extraction and signal cleaners.

    The class owns signal-side stage mechanics only. It can consume either a
    frame stream or a materialized frame buffer and returns a signal runtime
    state for analyzer execution.
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
    ) -> None:
        self._context = context
        self._stage_executor = stage_executor
        self._execution_policy = execution_policy
        self._lookahead = lookahead
        self._estimates = estimates
        self._resources = resources
        self._boundary_materializer = boundary_materializer

    def run(self, frames: FrameRuntimeState) -> SignalRuntimeState:
        """Run signal extraction and cleaners, preserving streaming when useful."""
        signal = self._run_signal_extractor(frames)
        return self._run_signal_cleaners(signal)

    def _run_signal_extractor(self, frames: FrameRuntimeState) -> SignalRuntimeState:
        decision = self._execution_policy.decide_stage(
            PipelineStagePolicyContext(
                stage_id="signal_extraction",
                stage_group="signal_extractor",
                stage_streamable=PipelineComponentCapabilities.can_stream_signal_extractor(
                    self._context.signal_extractor
                ),
                input_is_streaming=frames.is_streaming,
                downstream_streamable=self._lookahead.signal_successor_streamable(cleaner_index=0),
                estimated_queue_bytes=self._estimates.signal_queue_bytes,
                estimated_materialized_bytes=self._estimates.materialized_frame_bytes,
            )
        )
        if decision.streams:
            output = SignalBuffer(buffer_size=self._context.stream_runtime.signal_buffer_size)
            self._resources.signal_buffers.append(output)
            task = self._signal_extraction_task(frames.buffer, output)
            return SignalRuntimeState(
                buffer=output,
                pending_tasks=[*frames.pending_tasks, task],
                buffers=self._resources.signal_buffers,
            )

        frame_buffer = self._boundary_materializer.materialize_frames(
            frames,
            "signal_extraction.materialize_input",
        )
        signal = self._stage_executor.run(
            "signal_extraction",
            lambda: self._context.signal_extractor.extract(frame_buffer),
        )
        return SignalRuntimeState(signal=signal, buffers=self._resources.signal_buffers)

    def _run_signal_cleaners(self, state: SignalRuntimeState) -> SignalRuntimeState:
        for cleaner_index, cleaner in enumerate(self._context.signal_cleaners):
            decision = self._execution_policy.decide_stage(
                PipelineStagePolicyContext(
                    stage_id=f"signal_cleaning[{cleaner_index}]",
                    stage_group="signal_cleaners",
                    stage_streamable=PipelineComponentCapabilities.can_stream_signal_cleaner(cleaner),
                    input_is_streaming=state.is_streaming,
                    downstream_streamable=self._lookahead.signal_successor_streamable(
                        cleaner_index=cleaner_index + 1
                    ),
                    estimated_queue_bytes=self._estimates.signal_queue_bytes,
                )
            )
            if decision.streams:
                state = self._append_streaming_signal_cleaner(
                    state,
                    cleaner=cleaner,
                    cleaner_index=cleaner_index,
                )
                continue

            signal = self._boundary_materializer.materialize_signal(
                state,
                f"signal_cleaning[{cleaner_index}].materialize_input",
            )
            cleaned = self._stage_executor.run(
                f"signal_cleaning[{cleaner_index}]",
                lambda c=cleaner, s=signal: c.clean(s),
            )
            state = SignalRuntimeState(signal=cleaned, buffers=self._resources.signal_buffers)
        return state

    def _append_streaming_signal_cleaner(
        self,
        state: SignalRuntimeState,
        *,
        cleaner: Any,
        cleaner_index: int,
    ) -> SignalRuntimeState:
        input_state = self._boundary_materializer.ensure_signal_stream(state)
        output = SignalBuffer(buffer_size=self._context.stream_runtime.signal_buffer_size)
        self._resources.signal_buffers.append(output)
        input_state.buffer.set_consumer_count(1)
        subscription = input_state.buffer.subscribe(cleaner_index)
        task = self._signal_cleaner_task(
            subscription,
            output,
            cleaner=cleaner,
            cleaner_index=cleaner_index,
        )
        return SignalRuntimeState(
            buffer=output,
            pending_tasks=[*input_state.pending_tasks, task],
            buffers=self._resources.signal_buffers,
        )

    def _signal_extraction_task(
        self,
        frames,
        output: SignalBuffer,
    ) -> ThreadedStageTask:
        return lambda executor: executor.submit(
            lambda: self._stage_executor.run(
                "signal_extraction",
                lambda: self._context.signal_extractor.extract_into(frames, output),
            )
        )

    def _signal_cleaner_task(
        self,
        input_signal: Iterable[ISignalSample],
        output: SignalBuffer,
        *,
        cleaner: Any,
        cleaner_index: int,
    ) -> ThreadedStageTask:
        return lambda executor: executor.submit(
            lambda: self._stage_executor.run(
                f"signal_cleaning[{cleaner_index}]",
                lambda: cleaner.clean_into(input_signal, output),
            )
        )
