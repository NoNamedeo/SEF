from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any

from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.IData import IData
from library.core.interfaces.StreamingContracts import IStreamingSignalCleaner
from library.core.pipeline.FrameExporterExecutor import FrameExporterExecutor
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from library.core.pipeline.LatencyPolicy import FrameLatencyPolicy
from library.core.pipeline.PipelineBuffers import PipelineBuffers
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineExecutionResult import (
    FramePipelineResult,
    PipelineExecutionResult,
    ThreadedStageTask,
)
from library.core.pipeline.PipelineStageExecutor import PipelineStageExecutor
from library.core.pipeline.VisualizationExecutor import VisualizationExecutor
from library.core.pipeline.VisualizerBinding import VisualizerBinding
from library.core.visualization.VisualArtifact import VisualArtifact


def _stream_consumer_id(binding_index: int, result_index: int, result_count: int) -> int:
    """Return a stable per-visualizer consumer id."""
    return binding_index * max(result_count, 1) + result_index


class StreamingSignalTailExecutor:
    """
    Executes the concurrent streaming tail after frame processing.

    It wires frame exporters, signal extraction, signal cleaners, analyzers,
    and stream visualizers into one bounded-buffer graph. All graph-building
    rules live here, leaving ``Pipeline`` free from threading details.

    Concurrency model
    -----------------
    The executor starts consumers before producers:
    1. streaming visualizers subscribe to analyzer data buffers;
    2. streaming analyzers subscribe to the final signal buffer;
    3. signal cleaners subscribe to the previous signal buffer;
    4. signal extraction consumes the final frame stream;
    5. frame exporters and upstream frame tasks produce frame data.

    This order avoids bounded-buffer deadlocks because downstream consumers are
    already waiting before upstream stages publish data.

    Failure handling
    ----------------
    If any future fails, all known frame, signal, and data buffers are aborted.
    Aborting is required to unblock producers or consumers that may otherwise be
    waiting on a full or empty bounded buffer.
    """

    def __init__(
        self,
        *,
        context: PipelineContext,
        stage_executor: PipelineStageExecutor,
        frame_exporter_executor: FrameExporterExecutor,
        visualization_executor: VisualizationExecutor,
    ) -> None:
        self._context = context
        self._stage_executor = stage_executor
        self._frame_exporter_executor = frame_exporter_executor
        self._visualization_executor = visualization_executor

    def run(
        self,
        *,
        frame_pipeline: FramePipelineResult,
        intermediate_store: IntermediateFrameArtifactStore,
        latency_policy: FrameLatencyPolicy,
    ) -> PipelineExecutionResult:
        """
        Run every streaming-capable stage after the frame pipeline.

        Parameters
        ----------
        frame_pipeline:
            Output from ``FramePipelineExecutor`` containing the latest frame
            buffer and upstream frame tasks that still need to be scheduled.
        intermediate_store:
            Store containing intermediate frame captures produced upstream.
        latency_policy:
            Runtime latency policy whose metrics are attached to the final
            internal execution result.
        """
        artifacts: list[VisualArtifact] = []
        artifact_lock = Lock()
        signal_buffers: list[Any] = []
        data_buffers = [DataBuffer(buffer_size=self._context.stream_runtime.data_buffer_size) for _ in self._context.analyzers]
        results: list[IData | None] = [None] * len(self._context.analyzers)

        final_frame_buffer, frame_exporter_tasks = self._frame_exporter_executor.build_streaming_tasks(
            frame_pipeline.frame_buffer,
            frame_pipeline.frame_buffers,
            artifacts,
            artifact_lock,
        )
        final_signal_buffer, signal_cleaner_tasks = self._build_signal_cleaner_tasks(signal_buffers)
        visualizer_targets = self._visualization_executor.streaming_targets(len(data_buffers))
        self._configure_consumers(
            signal_buffers=signal_buffers,
            data_buffers=data_buffers,
            visualizer_targets=visualizer_targets,
        )

        self._run_concurrent_tail(
            final_frame_buffer=final_frame_buffer,
            first_signal_buffer=signal_buffers[0],
            final_signal_buffer=final_signal_buffer,
            pending_frame_tasks=frame_pipeline.pending_tasks,
            frame_exporter_tasks=frame_exporter_tasks,
            signal_cleaner_tasks=signal_cleaner_tasks,
            data_buffers=data_buffers,
            visualizer_targets=visualizer_targets,
            artifacts=artifacts,
            artifact_lock=artifact_lock,
            results=results,
            frame_buffers=frame_pipeline.frame_buffers,
            signal_buffers=signal_buffers,
        )

        intermediate_frames = intermediate_store.to_collection()
        debug_artifacts = self._visualization_executor.run_intermediate_visualizers(intermediate_frames)
        return PipelineExecutionResult(
            results=tuple(result for result in results if result is not None),
            final_artifacts=tuple(artifacts),
            debug_artifacts=tuple(debug_artifacts),
            intermediate_frames=intermediate_frames,
            latency_policy_metrics=latency_policy.metrics(),
        )

    def _build_signal_cleaner_tasks(
        self,
        signal_buffers: list[Any],
    ) -> tuple[Any, list[ThreadedStageTask]]:
        first_signal_buffer = SignalBuffer(buffer_size=self._context.stream_runtime.signal_buffer_size)
        signal_buffers.append(first_signal_buffer)
        current_signal = first_signal_buffer
        tasks: list[ThreadedStageTask] = []

        for cleaner_index, cleaner in enumerate(self._context.signal_cleaners):
            input_signal = current_signal.subscribe(cleaner_index)
            if not isinstance(cleaner, IStreamingSignalCleaner):
                raise TypeError(f"{type(cleaner).__name__} does not implement IStreamingSignalCleaner.")

            output_signal = SignalBuffer(buffer_size=self._context.stream_runtime.signal_buffer_size)
            signal_buffers.append(output_signal)
            tasks.append(
                self._signal_cleaner_task(
                    cleaner,
                    input_signal,
                    output_signal,
                    cleaner_index=cleaner_index,
                )
            )
            current_signal = output_signal

        return current_signal, tasks

    def _run_concurrent_tail(
        self,
        *,
        final_frame_buffer: FrameBuffer,
        first_signal_buffer: SignalBuffer,
        final_signal_buffer: Any,
        pending_frame_tasks: tuple[ThreadedStageTask, ...],
        frame_exporter_tasks: list[ThreadedStageTask],
        signal_cleaner_tasks: list[ThreadedStageTask],
        data_buffers: list[DataBuffer],
        visualizer_targets: list[tuple[int, VisualizerBinding, int]],
        artifacts: list[VisualArtifact],
        artifact_lock: Any,
        results: list[IData | None],
        frame_buffers: list[FrameBuffer],
        signal_buffers: list[Any],
    ) -> None:
        max_workers = max(
            1,
            len(pending_frame_tasks)
            + len(frame_exporter_tasks)
            + 1
            + len(signal_cleaner_tasks)
            + len(self._context.analyzers)
            + len(visualizer_targets),
        )
        futures: list[Future] = []
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sef-pipeline") as executor:
            futures.extend(
                self._submit_streaming_visualizers(
                    executor,
                    data_buffers=data_buffers,
                    visualizer_targets=visualizer_targets,
                    artifacts=artifacts,
                    artifact_lock=artifact_lock,
                )
            )
            futures.extend(
                self._submit_streaming_analyzers(
                    executor,
                    final_signal_buffer=final_signal_buffer,
                    data_buffers=data_buffers,
                    results=results,
                )
            )
            futures.extend(task(executor) for task in signal_cleaner_tasks)
            futures.append(self._submit_signal_extractor(executor, final_frame_buffer, first_signal_buffer))
            futures.extend(task(executor) for task in frame_exporter_tasks)
            futures.extend(task(executor) for task in pending_frame_tasks)

            try:
                for future in futures:
                    future.result()
            except Exception:
                PipelineBuffers.abort_all(frame_buffers, signal_buffers, data_buffers)
                raise

    def _configure_consumers(
        self,
        *,
        signal_buffers: list[Any],
        data_buffers: list[DataBuffer],
        visualizer_targets: list[tuple[int, VisualizerBinding, int]],
    ) -> None:
        """
        Declare subscriber counts before producers start publishing.

        ``SignalBuffer`` and ``DataBuffer`` need consumer counts to know when a
        published item may be released. Counts must be configured once, before
        any stage starts, otherwise a producer can publish data that never gets
        acknowledged by all expected consumers.
        """
        for signal_buffer in signal_buffers[:-1]:
            signal_buffer.set_consumer_count(1)
        signal_buffers[-1].set_consumer_count(len(self._context.analyzers))

        data_consumer_counts = [0] * len(data_buffers)
        for _, _, result_index in visualizer_targets:
            data_consumer_counts[result_index] += 1
        for data_buffer, consumer_count in zip(data_buffers, data_consumer_counts, strict=True):
            data_buffer.set_consumer_count(consumer_count)

    def _signal_cleaner_task(
        self,
        cleaner: IStreamingSignalCleaner,
        input_signal: Any,
        output_signal: SignalBuffer,
        *,
        cleaner_index: int,
    ) -> ThreadedStageTask:
        def submit_cleaner(executor: ThreadPoolExecutor) -> Future:
            return executor.submit(
                lambda: self._stage_executor.run(
                    f"signal_cleaning[{cleaner_index}]",
                    lambda: cleaner.clean_into(input_signal, output_signal),
                )
            )

        return submit_cleaner

    def _submit_signal_extractor(
        self,
        executor: ThreadPoolExecutor,
        final_frame_buffer: FrameBuffer,
        first_signal_buffer: SignalBuffer,
    ) -> Future:
        return executor.submit(
            lambda: self._stage_executor.run(
                "signal_extraction",
                lambda: self._context.signal_extractor.extract_into(final_frame_buffer, first_signal_buffer),
            )
        )

    def _submit_streaming_analyzers(
        self,
        executor: ThreadPoolExecutor,
        *,
        final_signal_buffer: Any,
        data_buffers: list[DataBuffer],
        results: list[IData | None],
    ) -> list[Future]:
        futures: list[Future] = []
        for analyzer_index, analyzer in enumerate(self._context.analyzers):
            signal_subscription = final_signal_buffer.subscribe(analyzer_index)
            futures.append(
                executor.submit(
                    lambda a=analyzer, s=signal_subscription, idx=analyzer_index: self._store_result(
                        results,
                        idx,
                        self._stage_executor.run(
                            f"analysis[{idx}]",
                            lambda: a.analyze_into(s, data_buffers[idx]),
                        ),
                    )
                )
            )
        return futures

    def _submit_streaming_visualizers(
        self,
        executor: ThreadPoolExecutor,
        *,
        data_buffers: list[DataBuffer],
        visualizer_targets: list[tuple[int, VisualizerBinding, int]],
        artifacts: list[VisualArtifact],
        artifact_lock: Any,
    ) -> list[Future]:
        futures: list[Future] = []
        result_count = len(data_buffers)
        for binding_index, binding, result_index in visualizer_targets:
            data_subscription = data_buffers[result_index].subscribe(
                _stream_consumer_id(binding_index, result_index, result_count)
            )
            futures.append(
                executor.submit(
                    lambda b=binding, idx=binding_index, ridx=result_index, data=data_subscription: self._extend_artifacts(
                        artifacts,
                        artifact_lock,
                        self._stage_executor.run(
                            f"visualisation[{idx}][{ridx}]",
                            lambda: b.visualizer.render_stream(data, self._visualization_executor.context_for(b, ridx, data)),
                        ),
                    )
                )
            )
        return futures

    @staticmethod
    def _store_result(results: list[IData | None], index: int, result: IData) -> None:
        results[index] = result

    @staticmethod
    def _extend_artifacts(
        artifacts: list[VisualArtifact],
        artifact_lock: Any,
        rendered: tuple[VisualArtifact, ...],
    ) -> None:
        with artifact_lock:
            artifacts.extend(rendered)
