from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor

from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.Signal import Signal
from library.core.interfaces.BufferContracts import ISubscribableBuffer
from library.core.interfaces.IData import IData
from library.core.interfaces.ISignal import ISignal
from library.core.pipeline.PipelineBoundaryMaterializer import PipelineBoundaryMaterializer
from library.core.pipeline.PipelineComponentCapabilities import PipelineComponentCapabilities
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineExecutionPolicy import (
    PipelineExecutionEstimates,
    PipelineExecutionPolicy,
    PipelineStagePolicyContext,
)
from library.core.pipeline.PipelineExecutionResources import PipelineExecutionResources
from library.core.pipeline.PipelineRuntimeState import SignalRuntimeState
from library.core.pipeline.PipelineStageExecutor import PipelineStageExecutor
from library.core.pipeline.VisualizationExecutor import VisualizationExecutor
from library.core.pipeline.VisualizerBinding import VisualizerBinding
from library.core.visualization.VisualArtifact import VisualArtifact

_SIGNAL_ANALYSIS_MATERIALIZER_CONSUMER_ID = -20_000


class AnalysisSegmentExecutor:
    """
    Executes analyzers and final visualizers.

    Analyzer execution is the only segment with fan-out: streaming analyzers,
    streaming visualizers and batch analyzers may all consume the same signal
    source through independent subscriptions.
    """

    def __init__(
        self,
        *,
        context: PipelineContext,
        stage_executor: PipelineStageExecutor,
        visualization_executor: VisualizationExecutor,
        execution_policy: PipelineExecutionPolicy,
        estimates: PipelineExecutionEstimates,
        resources: PipelineExecutionResources,
        boundary_materializer: PipelineBoundaryMaterializer,
    ) -> None:
        self._context = context
        self._stage_executor = stage_executor
        self._visualization_executor = visualization_executor
        self._execution_policy = execution_policy
        self._estimates = estimates
        self._resources = resources
        self._boundary_materializer = boundary_materializer

    def run(self, state: SignalRuntimeState) -> list[IData]:
        """Run analyzers and render final artifacts."""
        analyzer_modes = self._analyzer_streaming_modes(state)
        if not any(analyzer_modes):
            signal = self._boundary_materializer.materialize_signal(
                state,
                "analysis.materialize_input",
            )
            results = self._run_batch_analyzers(signal, range(len(self._context.analyzers)))
            self._resources.add_final_artifacts(
                self._visualization_executor.run_final_visualizers(results)
            )
            return results

        return self._run_mixed_analyzers(state, analyzer_modes)

    def _run_mixed_analyzers(
        self,
        state: SignalRuntimeState,
        analyzer_modes: list[bool],
    ) -> list[IData]:
        streaming_indexes = [index for index, streams in enumerate(analyzer_modes) if streams]
        batch_indexes = [index for index, streams in enumerate(analyzer_modes) if not streams]
        results: list[IData | None] = [None] * len(self._context.analyzers)
        final_signal = self._boundary_materializer.ensure_signal_stream(state)
        materialized_signal = state.signal

        visualizer_targets = self._streaming_visualizer_targets(set(streaming_indexes))
        main_thread_visualizer_targets = self._main_thread_visualizer_targets(visualizer_targets)
        threaded_visualizer_targets = [
            target for target in visualizer_targets if target not in main_thread_visualizer_targets
        ]
        streamed_target_keys = {
            (binding_index, result_index)
            for binding_index, _, result_index in visualizer_targets
        }
        data_buffers = self._data_buffers_for_streaming_analyzers(visualizer_targets)

        materializer_future: Future[ISignal] | None = None
        final_signal.buffer.set_consumer_count(
            len(streaming_indexes) + (1 if batch_indexes and materialized_signal is None else 0)
        )
        materializer_subscription = None
        if batch_indexes and materialized_signal is None:
            materializer_subscription = final_signal.buffer.subscribe(
                _SIGNAL_ANALYSIS_MATERIALIZER_CONSUMER_ID
            )

        with ThreadPoolExecutor(
            max_workers=self._max_analyzer_workers(
                final_signal,
                streaming_indexes,
                visualizer_targets,
                batch_indexes,
            )
        ) as executor:
            futures: list[Future] = []
            futures.extend(
                self._submit_streaming_visualizers(
                    executor,
                    visualizer_targets=threaded_visualizer_targets,
                    data_buffers=data_buffers,
                )
            )
            futures.extend(
                self._submit_streaming_analyzers(
                    executor,
                    final_signal=final_signal,
                    data_buffers=data_buffers,
                    results=results,
                    streaming_indexes=streaming_indexes,
                )
            )
            if materializer_subscription is not None:
                materializer_future = executor.submit(
                    lambda: self._stage_executor.run(
                        "analysis.materialize_signal_for_batch",
                        lambda: Signal(list(materializer_subscription)),
                    )
                )
                futures.append(materializer_future)
            futures.extend(task(executor) for task in final_signal.pending_tasks)

            try:
                self._run_main_thread_streaming_visualizers(
                    visualizer_targets=main_thread_visualizer_targets,
                    data_buffers=data_buffers,
                )
                for future in futures:
                    future.result()
            except Exception:
                self._resources.abort_all_buffers()
                raise

        if materializer_future is not None:
            materialized_signal = materializer_future.result()
        if batch_indexes:
            if materialized_signal is None:
                raise RuntimeError("Batch analyzers require a materialized signal.")
            batch_results = self._run_batch_analyzers(materialized_signal, batch_indexes)
            for index, result in zip(batch_indexes, batch_results, strict=True):
                results[index] = result

        final_results = [result for result in results if result is not None]
        self._resources.add_final_artifacts(
            self._visualization_executor.run_final_visualizers(
                final_results,
                skip_targets=streamed_target_keys,
            )
        )
        return final_results

    def _analyzer_streaming_modes(self, state: SignalRuntimeState) -> list[bool]:
        stream_visualizer_indexes = self._streaming_visualizer_result_indexes()
        return [
            self._execution_policy.decide_analyzer(
                PipelineStagePolicyContext(
                    stage_id=f"analysis[{index}]",
                    stage_group="analyzers",
                    stage_streamable=PipelineComponentCapabilities.can_stream_analyzer(analyzer),
                    input_is_streaming=state.is_streaming,
                    progressive_consumer=index in stream_visualizer_indexes,
                    estimated_queue_bytes=self._estimates.data_queue_bytes,
                )
            ).streams
            for index, analyzer in enumerate(self._context.analyzers)
        ]

    def _run_batch_analyzers(self, signal: ISignal, indexes: Iterable[int]) -> list[IData]:
        results: list[IData] = []
        for analyzer_index in indexes:
            analyzer = self._context.analyzers[analyzer_index]
            results.append(
                self._stage_executor.run(
                    f"analysis[{analyzer_index}]",
                    lambda a=analyzer: a.analyze(signal),
                )
            )
        return results

    def _streaming_visualizer_targets(
        self,
        streaming_analyzer_indexes: set[int],
    ) -> list[tuple[int, VisualizerBinding, int]]:
        return [
            target
            for target in self._visualization_executor.targets(len(self._context.analyzers))
            if target[2] in streaming_analyzer_indexes
            and PipelineComponentCapabilities.can_stream_visualizer(target[1].visualizer)
        ]

    def _streaming_visualizer_result_indexes(self) -> set[int]:
        return {
            result_index
            for _, binding, result_index in self._visualization_executor.targets(
                len(self._context.analyzers)
            )
            if PipelineComponentCapabilities.can_stream_visualizer(binding.visualizer)
        }

    @staticmethod
    def _main_thread_visualizer_targets(
        visualizer_targets: list[tuple[int, VisualizerBinding, int]],
    ) -> list[tuple[int, VisualizerBinding, int]]:
        return [
            target
            for target in visualizer_targets
            if bool(getattr(target[1].visualizer, "requires_main_thread", False))
        ]

    def _data_buffers_for_streaming_analyzers(
        self,
        visualizer_targets: list[tuple[int, VisualizerBinding, int]],
    ) -> list[ISubscribableBuffer[IData]]:
        consumer_counts = [0] * len(self._context.analyzers)
        for _, _, result_index in visualizer_targets:
            consumer_counts[result_index] += 1

        buffers: list[ISubscribableBuffer[IData]] = []
        for consumer_count in consumer_counts:
            buffer = DataBuffer(buffer_size=self._context.stream_runtime.data_buffer_size)
            buffer.set_consumer_count(consumer_count)
            buffers.append(buffer)
        self._resources.data_buffers.extend(buffers)
        return buffers

    def _submit_streaming_analyzers(
        self,
        executor: ThreadPoolExecutor,
        *,
        final_signal: SignalRuntimeState,
        data_buffers: list[ISubscribableBuffer[IData]],
        results: list[IData | None],
        streaming_indexes: list[int],
    ) -> list[Future]:
        futures: list[Future] = []
        for analyzer_index in streaming_indexes:
            analyzer = self._context.analyzers[analyzer_index]
            signal_subscription = final_signal.buffer.subscribe(analyzer_index)
            futures.append(
                executor.submit(
                    lambda a=analyzer, s=signal_subscription, idx=analyzer_index: (
                        self._store_result(
                            results,
                            idx,
                            self._stage_executor.run(
                                f"analysis[{idx}]",
                                lambda: a.analyze_into(s, data_buffers[idx]),
                            ),
                        )
                    )
                )
            )
        return futures

    def _submit_streaming_visualizers(
        self,
        executor: ThreadPoolExecutor,
        *,
        visualizer_targets: list[tuple[int, VisualizerBinding, int]],
        data_buffers: list[ISubscribableBuffer[IData]],
    ) -> list[Future]:
        futures: list[Future] = []
        result_count = len(data_buffers)
        for binding_index, binding, result_index in visualizer_targets:
            data_subscription = data_buffers[result_index].subscribe(
                self._stream_consumer_id(binding_index, result_index, result_count)
            )
            futures.append(
                executor.submit(
                    lambda b=binding, idx=binding_index, ridx=result_index, data=data_subscription: (
                        self._resources.add_final_artifacts(
                            self._render_streaming_visualizer(
                                binding=b,
                                binding_index=idx,
                                result_index=ridx,
                                data=data,
                            )
                        )
                    )
                )
            )
        return futures

    def _run_main_thread_streaming_visualizers(
        self,
        *,
        visualizer_targets: list[tuple[int, VisualizerBinding, int]],
        data_buffers: list[ISubscribableBuffer[IData]],
    ) -> None:
        result_count = len(data_buffers)
        for binding_index, binding, result_index in visualizer_targets:
            data_subscription = data_buffers[result_index].subscribe(
                self._stream_consumer_id(binding_index, result_index, result_count)
            )
            self._resources.add_final_artifacts(
                self._render_streaming_visualizer(
                    binding=binding,
                    binding_index=binding_index,
                    result_index=result_index,
                    data=data_subscription,
                )
            )

    def _render_streaming_visualizer(
        self,
        *,
        binding: VisualizerBinding,
        binding_index: int,
        result_index: int,
        data: Iterable[IData],
    ) -> tuple[VisualArtifact, ...]:
        return self._stage_executor.run(
            f"visualisation[{binding_index}][{result_index}]",
            lambda: binding.visualizer.render_stream(
                data,
                self._visualization_executor.context_for(binding, result_index, data),
            ),
        )

    @staticmethod
    def _max_analyzer_workers(
        final_signal: SignalRuntimeState,
        streaming_indexes: list[int],
        visualizer_targets: list[tuple[int, VisualizerBinding, int]],
        batch_indexes: list[int],
    ) -> int:
        materializer_count = 1 if batch_indexes and final_signal.signal is None else 0
        return max(
            1,
            len(final_signal.pending_tasks)
            + len(streaming_indexes)
            + len(visualizer_targets)
            + materializer_count,
        )

    @staticmethod
    def _stream_consumer_id(binding_index: int, result_index: int, result_count: int) -> int:
        return binding_index * max(result_count, 1) + result_index

    @staticmethod
    def _store_result(results: list[IData | None], index: int, result: IData) -> None:
        results[index] = result
