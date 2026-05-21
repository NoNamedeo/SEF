from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.BufferContracts import IBuffer
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalSample import ISignalSample
from library.core.pipeline.PipelineBuffers import PipelineBuffers
from library.core.pipeline.PipelineExecutionResources import PipelineExecutionResources
from library.core.pipeline.PipelineRuntimeState import FrameRuntimeState, SignalRuntimeState, ThreadedStageTask
from library.core.pipeline.PipelineStageExecutor import PipelineStageExecutor

_SIGNAL_MATERIALIZER_CONSUMER_ID = -10_000


class PipelineBoundaryMaterializer:
    """
    Converts active streams into replayable batch values at explicit boundaries.

    The component centralizes the thread orchestration needed to drain pending
    producers while materializing their output. Segment executors use it instead
    of owning generic boundary mechanics themselves.
    """

    def __init__(
        self,
        *,
        stage_executor: PipelineStageExecutor,
        resources: PipelineExecutionResources,
        signal_buffer_size: int,
    ) -> None:
        self._stage_executor = stage_executor
        self._resources = resources
        self._signal_buffer_size = signal_buffer_size

    def materialize_frames(self, state: FrameRuntimeState, stage_name: str) -> FrameBuffer:
        """Return a replayable frame buffer, draining pending stream tasks if needed."""
        if not state.pending_tasks:
            return state.buffer

        with ThreadPoolExecutor(
            max_workers=len(state.pending_tasks) + 1,
            thread_name_prefix="sef-frame-boundary",
        ) as executor:
            materialized_future = executor.submit(
                lambda: self._stage_executor.run(
                    stage_name,
                    lambda: PipelineBuffers.copy_frame_buffer(state.buffer),
                )
            )
            futures = [materialized_future, *self._submit_pending(executor, state.pending_tasks)]
            try:
                for future in futures:
                    future.result()
            except Exception:
                self._resources.abort_all_buffers()
                raise
        return materialized_future.result()

    def materialize_signal(self, state: SignalRuntimeState, stage_name: str) -> ISignal:
        """Return a materialized signal, draining pending signal tasks if needed."""
        if state.signal is not None:
            return state.signal
        if state.buffer is None:
            raise RuntimeError("Cannot materialize an empty signal state.")

        state.buffer.set_consumer_count(1)
        subscription = state.buffer.subscribe(_SIGNAL_MATERIALIZER_CONSUMER_ID)
        with ThreadPoolExecutor(
            max_workers=len(state.pending_tasks) + 1,
            thread_name_prefix="sef-signal-boundary",
        ) as executor:
            materialized_future = executor.submit(
                lambda: self._stage_executor.run(stage_name, lambda: Signal(list(subscription)))
            )
            futures = [materialized_future, *self._submit_pending(executor, state.pending_tasks)]
            try:
                for future in futures:
                    future.result()
            except Exception:
                self._resources.abort_all_buffers()
                raise
        return materialized_future.result()

    def ensure_signal_stream(self, state: SignalRuntimeState) -> SignalRuntimeState:
        """Return a streaming signal state, re-publishing materialized input if needed."""
        if state.buffer is not None:
            return state
        if state.signal is None:
            raise RuntimeError("Cannot stream an empty signal state.")

        output = SignalBuffer(buffer_size=self._signal_buffer_size)
        self._resources.signal_buffers.append(output)
        return SignalRuntimeState(
            signal=state.signal,
            buffer=output,
            pending_tasks=[self._signal_publisher_task(state.signal, output)],
            buffers=self._resources.signal_buffers,
        )

    def _signal_publisher_task(self, signal: ISignal, output: IBuffer[ISignalSample]) -> ThreadedStageTask:
        return lambda executor: executor.submit(
            lambda: self._stage_executor.run(
                "signal_stream.publish",
                lambda: self._publish_signal(signal, output),
            )
        )

    @staticmethod
    def _publish_signal(signal: ISignal, output: IBuffer[ISignalSample]) -> None:
        try:
            for sample in signal:
                output.put(sample)
        finally:
            output.close()

    @staticmethod
    def _submit_pending(
        executor: ThreadPoolExecutor,
        tasks: list[ThreadedStageTask],
    ):
        return [task(executor) for task in tasks]
