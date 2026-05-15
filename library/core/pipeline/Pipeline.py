from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Mapping

from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.IntermediateFrameArtifacts import IntermediateFrameArtifactCollection
from library.core.interfaces.IData import IData
from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.IFrameExporter import FrameExportContext
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.pipeline.FrameProcessingStage import FrameProcessorExecutionContext
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.VisualizerBinding import VisualizerBinding
from library.core.visualization.PipelineOutputs import PipelineOutputs
from library.core.visualization.PipelineRunMetadata import PipelineRunMetadata
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext

log = logging.getLogger(__name__)


def _stream_consumer_id(binding_index: int, result_index: int, result_count: int) -> int:
    """Return a stable per-visualizer consumer id."""
    return binding_index * max(result_count, 1) + result_index


class Pipeline:
    """
    Pure execution unit — the 'dumb worker' of the system.

    Design rationale
    ----------------
    Pipeline knows NOTHING about how it was built, which plugin was chosen,
    or what the data means.  Its only job is to walk the steps declared in
    PipelineContext in the correct order and return results.

    This strict separation means:
    - Orchestrators can swap contexts without touching execution logic.
    - Execution can be tested by injecting a mock context.
    - Future async / parallel variants only need to subclass or wrap this
      class, leaving all orchestration logic untouched.

    Execution order
    ---------------
    1. Frame extraction   (frame_extractor  → raw buffer)
    2. Frame processing   (frame_processors → processed buffer) [optional]
    3. Signal extraction  (signal_extractor → raw signal)
    4. Signal cleaning    (signal_cleaners  → smoothed signal)  [optional]
    5. Analysis           (analyzers        → list[IData])
    6. Visualisation      (visualizers)                         [optional]

    Event injection
    ---------------
    Pipeline inspects every component in the context: those that implement
    ``IEventEmitter`` get the current event bus and execution metadata
    injected **before** execution begins. This allows components to emit
    domain events during their work without any coupling to the Orchestrator.

    Raises
    ------
    PipelineExecutionError
        Wraps any exception raised by a pipeline step, enriching it with
        the name of the failing stage so callers can act accordingly.
    """

    def __init__(
        self,
        context: PipelineContext,
        event_bus: IEventBus | None = None,
        pipeline_id: str | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._context = context
        self._event_bus = event_bus
        self._pipeline_id = pipeline_id
        self._execution_metadata = dict(execution_metadata or {})

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> PipelineOutputs:
        """Execute the full pipeline and return results plus visual artifacts."""
        self._inject_event_bus(self._event_bus)
        return self._run_adaptive()

    # ── Internals ───────────────────────────────────────────────────────────

    def _run_adaptive(self) -> PipelineOutputs:
        """Run one adaptive pipeline, materializing only at non-streaming frame boundaries."""
        ctx = self._context
        intermediate_store = IntermediateFrameArtifactStore(ctx.intermediate_frame_capture)
        frame_buffers: list[FrameBuffer] = []
        artifacts: list[VisualArtifact] = []
        artifact_lock = Lock()

        frame_buffer, frame_tasks = self._build_adaptive_frame_pipeline(
            intermediate_store=intermediate_store,
            frame_buffers=frame_buffers,
        )
        exporters_streaming = all(callable(getattr(exporter, "export_into", None)) for exporter in ctx.frame_exporters)

        if self._can_stream_signal_pipeline() and exporters_streaming:
            outputs = self._run_streaming_signal_tail(
                frame_buffer=frame_buffer,
                pending_frame_tasks=frame_tasks,
                frame_buffers=frame_buffers,
                artifacts=artifacts,
                artifact_lock=artifact_lock,
                intermediate_store=intermediate_store,
            )
            return self._with_reproducibility_exports(outputs)

        materialized_buffer = self._materialize_pending_frame_stream(
            frame_buffer,
            frame_tasks,
            frame_buffers=frame_buffers,
            stage_name="frame_processing.materialize_final",
        )
        materialized_buffer, frame_export_artifacts = self._run_frame_exporters(materialized_buffer)
        artifacts.extend(frame_export_artifacts)

        signal = self._run_step("signal_extraction", lambda: ctx.signal_extractor.extract(materialized_buffer))
        for i, cleaner in enumerate(ctx.signal_cleaners):
            signal = self._run_step(
                f"signal_cleaning[{i}]",
                lambda c=cleaner: c.clean(signal),  # noqa: B023
            )

        results: list[IData] = []
        for i, analyzer in enumerate(ctx.analyzers):
            data = self._run_step(f"analysis[{i}]", lambda a=analyzer: a.analyze(signal))
            results.append(data)

        intermediate_frames = intermediate_store.to_collection()
        final_artifacts = [*artifacts, *self._run_visualizers(results)]
        debug_artifacts = self._run_intermediate_frame_visualizers(intermediate_frames)
        return self._with_reproducibility_exports(
            self._build_outputs(
                results=tuple(results),
                final_artifacts=tuple(final_artifacts),
                debug_artifacts=tuple(debug_artifacts),
                intermediate_frames=intermediate_frames,
            )
        )

    def _run_streaming_signal_tail(
        self,
        *,
        frame_buffer: FrameBuffer,
        pending_frame_tasks: list[Callable[[ThreadPoolExecutor], Future]],
        frame_buffers: list[FrameBuffer],
        artifacts: list[VisualArtifact],
        artifact_lock: Lock,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> PipelineOutputs:
        """Run frame exporters and signal/result stages concurrently after adaptive frame processing."""
        ctx = self._context
        signal_buffers: list[Any] = []
        data_buffers = [analyzer.buffer for analyzer in ctx.analyzers]
        results: list[IData | None] = [None] * len(ctx.analyzers)

        final_frame_buffer, frame_exporter_tasks = self._build_streaming_frame_export_tasks(
            frame_buffer,
            frame_buffers,
            artifacts,
            artifact_lock,
        )

        first_signal_buffer = ctx.signal_extractor.buffer
        signal_buffers.append(first_signal_buffer)
        final_signal_buffer, signal_cleaner_tasks = self._build_streaming_signal_cleaner_tasks(
            first_signal_buffer,
            signal_buffers,
        )

        visualizer_targets = self._streaming_visualizer_targets(len(data_buffers))
        self._configure_streaming_consumers(
            signal_buffers=signal_buffers,
            data_buffers=data_buffers,
            visualizer_targets=visualizer_targets,
        )

        max_workers = max(
            1,
            len(pending_frame_tasks)
            + len(frame_exporter_tasks)
            + 1
            + len(signal_cleaner_tasks)
            + len(ctx.analyzers)
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
                    results=results,
                )
            )
            futures.extend(task(executor) for task in signal_cleaner_tasks)
            futures.append(
                executor.submit(
                    lambda: self._run_step(
                        "signal_extraction",
                        lambda: ctx.signal_extractor.extract(final_frame_buffer),
                    )
                )
            )
            futures.extend(task(executor) for task in frame_exporter_tasks)
            futures.extend(task(executor) for task in pending_frame_tasks)

            try:
                for future in futures:
                    future.result()
            except Exception:
                self._abort_streaming_buffers(frame_buffers, signal_buffers, data_buffers)
                raise

        intermediate_frames = intermediate_store.to_collection()
        debug_artifacts = self._run_intermediate_frame_visualizers(intermediate_frames)
        return self._build_outputs(
            results=tuple(result for result in results if result is not None),
            final_artifacts=tuple(artifacts),
            debug_artifacts=tuple(debug_artifacts),
            intermediate_frames=intermediate_frames,
        )

    def _build_adaptive_frame_pipeline(
        self,
        *,
        intermediate_store: IntermediateFrameArtifactStore,
        frame_buffers: list[FrameBuffer],
    ) -> tuple[FrameBuffer, list[Callable[[ThreadPoolExecutor], Future]]]:
        ctx = self._context
        source_buffer = getattr(ctx.frame_extractor, "buffer", None)
        if isinstance(source_buffer, FrameBuffer):
            frame_buffers.append(source_buffer)
            current_buffer = source_buffer
            pending_tasks = [self._frame_extraction_task()]
        else:
            current_buffer = self._run_step("frame_extraction", lambda: ctx.frame_extractor.extract())
            frame_buffers.append(current_buffer)
            pending_tasks: list[Callable[[ThreadPoolExecutor], Future]] = []

        for processor_index, processor in enumerate(ctx.frame_processors):
            if self._can_stream_frame_processor(processor):
                output_buffer = current_buffer.clone_empty()
                frame_buffers.append(output_buffer)
                pending_tasks.append(
                    self._frame_processor_task(
                        current_buffer,
                        output_buffer,
                        processor=processor,
                        processor_index=processor_index,
                        intermediate_store=intermediate_store,
                    )
                )
                current_buffer = output_buffer
                continue

            current_buffer = self._materialize_pending_frame_stream(
                current_buffer,
                pending_tasks,
                frame_buffers=frame_buffers,
                stage_name=f"frame_processing[{processor_index}].materialize_input",
            )
            pending_tasks = []
            current_buffer = self._process_complete_sequence_processor(
                current_buffer,
                processor,
                processor_index=processor_index,
                intermediate_store=intermediate_store,
            )
            frame_buffers.append(current_buffer)

        return current_buffer, pending_tasks

    def _materialize_pending_frame_stream(
        self,
        source_buffer: FrameBuffer,
        pending_tasks: list[Callable[[ThreadPoolExecutor], Future]],
        *,
        frame_buffers: list[FrameBuffer],
        stage_name: str,
    ) -> FrameBuffer:
        if not pending_tasks:
            return self._run_step(stage_name, lambda: self._copy_frame_buffer(source_buffer))

        with ThreadPoolExecutor(max_workers=len(pending_tasks) + 1, thread_name_prefix="sef-frame-boundary") as executor:
            futures = [task(executor) for task in pending_tasks]
            materialized_future = executor.submit(
                lambda: self._run_step(stage_name, lambda: self._copy_frame_buffer(source_buffer))
            )
            futures.append(materialized_future)
            try:
                for future in futures:
                    future.result()
            except Exception:
                self._abort_streaming_buffers(frame_buffers, [], [])
                raise
        return materialized_future.result()

    @staticmethod
    def _copy_frame_buffer(source_buffer: FrameBuffer) -> FrameBuffer:
        frames = list(source_buffer)
        output = FrameBuffer(buffer_size=max(len(frames) + 1, source_buffer.capacity))
        for frame in frames:
            output.put(frame)
        output.close()
        return output

    def _process_complete_sequence_processor(
        self,
        buffer: FrameBuffer,
        processor: Any,
        *,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> FrameBuffer:
        return self._run_step(
            f"frame_processing[{processor_index}]",
            lambda: self._process_frame_buffer_processor(
                buffer,
                processor,
                processor_index=processor_index,
                intermediate_store=intermediate_store,
            ),
        )

    @staticmethod
    def _process_frame_buffer_processor(
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

    def _frame_extraction_task(self) -> Callable[[ThreadPoolExecutor], Future]:
        return lambda executor: executor.submit(
            lambda: self._run_step("frame_extraction", lambda: self._context.frame_extractor.extract())
        )

    def _frame_processor_task(
        self,
        input_buffer: FrameBuffer,
        output_buffer: FrameBuffer,
        *,
        processor: Any,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> Callable[[ThreadPoolExecutor], Future]:
        def submit_processor(executor: ThreadPoolExecutor) -> Future:
            return executor.submit(
                lambda: self._run_step(
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

    @staticmethod
    def _can_stream_frame_processor(processor: Any) -> bool:
        capabilities = getattr(processor, "capabilities", None)
        return (
            bool(getattr(capabilities, "supports_streaming", False))
            and not bool(getattr(capabilities, "requires_complete_sequence", True))
            and callable(getattr(processor, "process_into", None))
        )

    def _can_stream_signal_pipeline(self) -> bool:
        ctx = self._context
        return (
            hasattr(ctx.signal_extractor, "buffer")
            and all(hasattr(cleaner, "buffer") for cleaner in ctx.signal_cleaners)
            and all(hasattr(analyzer, "buffer") for analyzer in ctx.analyzers)
        )

    def _build_streaming_frame_export_tasks(
        self,
        source_buffer: FrameBuffer,
        frame_buffers: list[FrameBuffer],
        artifacts: list[VisualArtifact],
        artifact_lock: Lock,
    ) -> tuple[FrameBuffer, list[Callable[[ThreadPoolExecutor], Future]]]:
        current_buffer = source_buffer
        tasks: list[Callable[[ThreadPoolExecutor], Future]] = []
        for exporter_index, exporter in enumerate(self._context.frame_exporters):
            output_buffer = current_buffer.clone_empty()
            frame_buffers.append(output_buffer)

            def submit_exporter(
                executor: ThreadPoolExecutor,
                *,
                stage_input: FrameBuffer = current_buffer,
                stage_output: FrameBuffer = output_buffer,
                frame_exporter: Any = exporter,
                index: int = exporter_index,
            ) -> Future:
                return executor.submit(
                    lambda: self._extend_streaming_artifacts(
                        artifacts,
                        artifact_lock,
                        self._run_step(
                            f"frame_export[{index}]",
                            lambda: frame_exporter.export_into(
                                stage_input,
                                stage_output,
                                FrameExportContext(
                                    pipeline_id=self._pipeline_id,
                                    exporter_name=type(frame_exporter).__name__,
                                    execution_metadata=dict(self._execution_metadata),
                                ),
                            ),
                        ),
                    )
                )

            tasks.append(submit_exporter)
            current_buffer = output_buffer
        return current_buffer, tasks

    def _build_streaming_signal_cleaner_tasks(
        self,
        first_signal_buffer: Any,
        signal_buffers: list[Any],
    ) -> tuple[Any, list[Callable[[ThreadPoolExecutor], Future]]]:
        current_signal = first_signal_buffer
        tasks: list[Callable[[ThreadPoolExecutor], Future]] = []
        for cleaner_index, cleaner in enumerate(self._context.signal_cleaners):
            input_signal = current_signal.subscribe(cleaner_index)
            output_signal = cleaner.buffer
            signal_buffers.append(output_signal)

            def submit_cleaner(
                executor: ThreadPoolExecutor,
                *,
                cleaner_component: Any = cleaner,
                signal_input: Any = input_signal,
                index: int = cleaner_index,
            ) -> Future:
                return executor.submit(
                    lambda: self._run_step(
                        f"signal_cleaning[{index}]",
                        lambda: cleaner_component.clean(signal_input),
                    )
                )

            tasks.append(submit_cleaner)
            current_signal = output_signal
        return current_signal, tasks

    def _configure_streaming_consumers(
        self,
        *,
        signal_buffers: list[Any],
        data_buffers: list[DataBuffer],
        visualizer_targets: list[tuple[int, VisualizerBinding, int]],
    ) -> None:
        for signal_buffer in signal_buffers[:-1]:
            signal_buffer.set_consumer_count(1)
        signal_buffers[-1].set_consumer_count(len(self._context.analyzers))

        data_consumer_counts = [0] * len(data_buffers)
        for _, _, result_index in visualizer_targets:
            data_consumer_counts[result_index] += 1
        for data_buffer, consumer_count in zip(data_buffers, data_consumer_counts, strict=True):
            data_buffer.set_consumer_count(consumer_count)

    def _streaming_visualizer_targets(
        self,
        result_count: int,
    ) -> list[tuple[int, VisualizerBinding, int]]:
        bindings = [
            *(VisualizerBinding(visualizer) for visualizer in self._context.visualizers),
            *self._context.visualizer_bindings,
        ]
        targets: list[tuple[int, VisualizerBinding, int]] = []
        for binding_index, binding in enumerate(bindings):
            target_indexes = self._run_step(
                f"visualisation[{binding_index}].targets",
                lambda b=binding: self._resolve_visualizer_targets(b, result_count),
            )
            for result_index in target_indexes:
                targets.append((binding_index, binding, result_index))
        return targets

    def _submit_streaming_analyzers(
        self,
        executor: ThreadPoolExecutor,
        *,
        final_signal_buffer: Any,
        results: list[Any],
    ) -> list[Future]:
        futures: list[Future] = []
        for analyzer_index, analyzer in enumerate(self._context.analyzers):
            signal_subscription = final_signal_buffer.subscribe(analyzer_index)
            futures.append(
                executor.submit(
                    lambda a=analyzer, s=signal_subscription, idx=analyzer_index: self._store_streaming_result(
                        results,
                        idx,
                        self._run_step(
                            f"analysis[{idx}]",
                            lambda: a.analyze(s),
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
        artifact_lock: Lock,
    ) -> list[Future]:
        futures: list[Future] = []
        result_count = len(data_buffers)
        for binding_index, binding, result_index in visualizer_targets:
            data_subscription = data_buffers[result_index].subscribe(
                _stream_consumer_id(binding_index, result_index, result_count)
            )
            futures.append(
                executor.submit(
                    lambda b=binding, idx=binding_index, ridx=result_index, data=data_subscription: self._extend_streaming_artifacts(
                        artifacts,
                        artifact_lock,
                        self._run_step(
                            f"visualisation[{idx}][{ridx}]",
                            lambda: b.visualizer.render(
                                data,
                                self._visualization_context(
                                    b,
                                    ridx,
                                    data,
                                ),
                            ),
                        ),
                    )
                )
            )
        return futures

    @staticmethod
    def _store_streaming_result(results: list[Any], index: int, result: Any) -> None:
        results[index] = result

    @staticmethod
    def _extend_streaming_artifacts(
        artifacts: list[VisualArtifact],
        artifact_lock: Lock,
        rendered: tuple[VisualArtifact, ...],
    ) -> None:
        with artifact_lock:
            artifacts.extend(rendered)

    @staticmethod
    def _abort_streaming_buffers(
        frame_buffers: list[FrameBuffer],
        signal_buffers: list[Any],
        data_buffers: list[DataBuffer],
    ) -> None:
        for buffer in [*frame_buffers, *signal_buffers, *data_buffers]:
            abort = getattr(buffer, "abort", None)
            if callable(abort):
                abort()

    def _build_outputs(
        self,
        *,
        results: tuple[Any, ...],
        final_artifacts: tuple[VisualArtifact, ...],
        debug_artifacts: tuple[VisualArtifact, ...],
        intermediate_frames: IntermediateFrameArtifactCollection,
    ) -> PipelineOutputs:
        return PipelineOutputs(
            results=results,
            final_artifacts=final_artifacts,
            debug_artifacts=debug_artifacts,
            metadata=PipelineRunMetadata(
                pipeline_id=self._resolved_pipeline_id(),
                generated_at=datetime.now(timezone.utc),
                execution_metadata=dict(self._execution_metadata),
            ),
            intermediate_frames=intermediate_frames,
        )

    def _inject_event_bus(self, bus: IEventBus | None) -> None:
        """
        Walk every component in the context and inject event dependencies
        into those that implement IEventEmitter.

        Called once at the beginning of ``run()``. The bus may be None; in
        that case emitters are explicitly reset to silent no-op mode.
        """
        components: list[Any] = [
            self._context.frame_extractor,
            self._context.signal_extractor,
            *self._context.frame_processors,
            *self._context.frame_exporters,
            *self._context.signal_cleaners,
            *self._context.analyzers,
            *self._context.visualizers,
            *(binding.visualizer for binding in self._context.visualizer_bindings),
            *self._context.intermediate_frame_visualizers,
        ]
        for component in components:
            if isinstance(component, IEventEmitter):
                component.event_bus = bus
                component.event_metadata = self._event_metadata()
                log.debug(
                    "Injected event context into %s",
                    type(component).__name__,
                )

    def _event_metadata(self) -> dict[str, str]:
        return self._runtime_metadata()

    def _runtime_metadata(self) -> dict[str, Any]:
        metadata = dict(self._execution_metadata)
        if self._pipeline_id is not None:
            metadata.setdefault("pipeline_id", self._pipeline_id)
        return metadata

    def _resolved_pipeline_id(self) -> str:
        return self._pipeline_id or "pipeline-unknown"

    def _with_reproducibility_exports(self, outputs: PipelineOutputs) -> PipelineOutputs:
        """
        Attach shareable config/code exports after artifacts and metadata exist.

        Exporters live in the core pipeline layer and do not depend on UI
        packages, so executed pipelines are reproducible in CLI, tests, and
        Streamlit alike.
        """
        from library.core.pipeline.PipelineCodeExporter import PipelineCodeExporter
        from library.core.pipeline.PipelineConfigExporter import PipelineConfigExporter

        config_exporter = PipelineConfigExporter()
        export_config = config_exporter.export(self._context, outputs)
        reproducibility = {
            "config": export_config,
            "json": config_exporter.to_json(export_config),
            "yaml": config_exporter.to_yaml(export_config),
            "python_builder_code": PipelineCodeExporter().export_config(export_config),
        }
        return PipelineOutputs(
            results=outputs.results,
            final_artifacts=outputs.final_artifacts,
            debug_artifacts=outputs.debug_artifacts,
            metadata=PipelineRunMetadata(
                pipeline_id=outputs.metadata.pipeline_id,
                generated_at=outputs.metadata.generated_at,
                execution_metadata=outputs.metadata.execution_metadata,
                reproducibility=reproducibility,
            ),
            intermediate_frames=outputs.intermediate_frames,
        )

    def _run_frame_exporters(self, buffer):
        artifacts: list[VisualArtifact] = []
        current_buffer = buffer
        for exporter_index, exporter in enumerate(self._context.frame_exporters):
            result = self._run_step(
                f"frame_export[{exporter_index}]",
                lambda e=exporter, b=current_buffer: e.export(
                    b,
                    FrameExportContext(
                        pipeline_id=self._pipeline_id,
                        exporter_name=type(e).__name__,
                        execution_metadata=dict(self._execution_metadata),
                    ),
                ),
            )
            current_buffer = result.buffer
            artifacts.extend(result.artifacts)
        return current_buffer, artifacts

    def _run_visualizers(self, results: list[IData]) -> list[VisualArtifact]:
        bindings = [
            *(VisualizerBinding(visualizer) for visualizer in self._context.visualizers),
            *self._context.visualizer_bindings,
        ]
        artifacts: list[VisualArtifact] = []

        for binding_index, binding in enumerate(bindings):
            target_indexes = self._run_step(
                f"visualisation[{binding_index}].targets",
                lambda b=binding: self._resolve_visualizer_targets(b, len(results)),
            )
            for result_index in target_indexes:
                data = results[result_index]
                rendered = self._run_step(
                    f"visualisation[{binding_index}][{result_index}]",
                    lambda v=binding.visualizer, d=data, ctx=self._visualization_context(binding, result_index, data): v.render(d, ctx),  # noqa: B008
                )
                artifacts.extend(rendered)
        return artifacts

    def _run_intermediate_frame_visualizers(
        self,
        intermediate_frames: IntermediateFrameArtifactCollection,
    ) -> list[VisualArtifact]:
        if intermediate_frames.is_empty:
            return []

        artifacts: list[VisualArtifact] = []
        for visualizer_index, visualizer in enumerate(self._context.intermediate_frame_visualizers):
            rendered = self._run_step(
                f"visualisation.intermediate_frames[{visualizer_index}]",
                lambda v=visualizer: v.render(
                    intermediate_frames,
                    self._intermediate_frame_visualization_context(v, intermediate_frames),
                ),
            )
            artifacts.extend(rendered)
        return artifacts

    def _intermediate_frame_visualization_context(
        self,
        visualizer: Any,
        intermediate_frames: IntermediateFrameArtifactCollection,
    ) -> VisualizationContext:
        return VisualizationContext(
            pipeline_id=self._pipeline_id,
            visualizer_name=type(visualizer).__name__,
            source_metadata=dict(intermediate_frames.metadata),
            execution_metadata=dict(self._execution_metadata),
            render_hints={"source": "intermediate_frames"},
        )

    def _visualization_context(
        self,
        binding: VisualizerBinding,
        result_index: int,
        data: IData,
    ) -> VisualizationContext:
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

    @staticmethod
    def _resolve_visualizer_targets(
        binding: VisualizerBinding,
        result_count: int,
    ) -> tuple[int, ...]:
        if binding.result_indices is None:
            return tuple(range(result_count))
        invalid = [index for index in binding.result_indices if index >= result_count]
        if invalid:
            raise ValueError(f"Visualizer target index out of range: {invalid}; available result indexes: 0..{result_count - 1}.")
        return tuple(binding.result_indices)

    @staticmethod
    def _run_step(stage: str, fn: Callable[[], Any]) -> Any:
        """
        Execute *fn* and wrap any exception with stage information.

        This method is intentionally kept trivial so subclasses can
        override it to add timing, retries, or async execution without
        altering the run() logic.
        """
        log.debug("Pipeline stage starting: %s", stage)
        try:
            result = fn()
            log.debug("Pipeline stage completed: %s", stage)
            return result
        except Exception as exc:
            log.error("Pipeline stage FAILED: %s — %s", stage, exc, exc_info=True)
            raise PipelineExecutionError(stage, exc) from exc


class PipelineExecutionError(RuntimeError):
    """
    Raised when a pipeline step fails.

    Attributes
    ----------
    stage : str
        Name of the failing stage (e.g. 'signal_extraction').
    cause : Exception
        The original exception raised by the step.
    """

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"Pipeline failed at stage '{stage}': {cause}")
        self.stage = stage
        self.cause = cause
