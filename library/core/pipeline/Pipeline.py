from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.IntermediateFrameArtifacts import IntermediateFrameArtifactCollection
from library.core.interfaces.IData import IData
from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.IFrameExporter import FrameExportContext
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.pipeline.FrameProcessingStage import FrameProcessingStage
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.VisualizerBinding import VisualizerBinding
from library.core.visualization.PipelineOutputs import PipelineOutputs
from library.core.visualization.PipelineRunMetadata import PipelineRunMetadata
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext

log = logging.getLogger(__name__)


def _cantor(n, m):
    return (((n + m)(n + m + 1)) / 2) + m


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
        self._frame_processing_stage = FrameProcessingStage()

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> PipelineOutputs:

        self._inject_event_bus(self._event_bus)

        ctx = self._context

        frame_buffer = ctx.frame_extractor.buffer

        extractor_thread = threading.Thread(
            target=lambda: self._run_step(
                "frame_extraction",
                lambda: ctx.frame_extractor.extract(),
            ),
            daemon=True,
        )

        extractor_thread.start()

        # TODO aggiungi anche i frame intermedi
        intermediate_store = IntermediateFrameArtifactStore(ctx.intermediate_frame_capture)
        buffer = self._run_step(
            "frame_processing",
            lambda: self._frame_processing_stage.apply(
                buffer,
                ctx.frame_processors,
                intermediate_store=intermediate_store,
            ),
        )
        intermediate_frames = intermediate_store.to_collection()
        buffer, frame_export_artifacts = self._run_frame_exporters(buffer)
        # TODO fine parte da modificare su frame intermedi

        signal_buffer = ctx.signal_extractor.buffer

        signal_thread = threading.Thread(
            target=lambda: self._run_step(
                "signal_extraction",
                lambda: ctx.signal_extractor.extract(frame_buffer),
            ),
            daemon=True,
        )

        signal_thread.start()

        signal_cleaners_number = len(ctx.signal_cleaners)
        analyzers_number = len(ctx.analyzers)

        cleaner_threads = []

        current_signal = signal_buffer

        if signal_cleaners_number == 0:
            current_signal._consumers_default = analyzers_number
        else:
            for i, cleaner in enumerate(ctx.signal_cleaners):
                input_signal = current_signal.subscribe(i)
                output_signal = cleaner.buffer

                thread = threading.Thread(
                    target=lambda c=cleaner, s=input_signal, idx=i: self._run_step(
                        f"signal_cleaning[{idx}]",
                        lambda: c.clean(s),
                    ),
                    daemon=True,
                )

                thread.start()

                cleaner_threads.append(thread)

                current_signal = output_signal

            current_signal._consumers_default = analyzers_number

        results: list[IData] = []

        analyzer_threads = []

        data_buffers: list[DataBuffer] = []

        for i, analyzer in enumerate(ctx.analyzers):
            signal_subscription = current_signal.subscribe(i)

            data_buffers.append(analyzer.buffer)

            thread = threading.Thread(
                target=lambda a=analyzer, s=signal_subscription, idx=i: results.append(
                    self._run_step(
                        f"analysis[{idx}]",
                        lambda: a.analyze(s),
                    )
                ),
                daemon=True,
            )

            thread.start()

            analyzer_threads.append(thread)

        visual_artifacts: list[VisualArtifact] = []

        visualizer_threads = self._run_visualizers_streaming(
            data_buffers=data_buffers,
            artifacts=visual_artifacts,
        )

        extractor_thread.join()
        signal_thread.join()

        for t in cleaner_threads:
            t.join()

        for t in analyzer_threads:
            t.join()

        for t in visualizer_threads:
            t.join()

        #TODO
        final_artifacts = [*frame_export_artifacts, *self._run_visualizers(results)]
        debug_artifacts = self._run_intermediate_frame_visualizers(intermediate_frames)
        #TODO
        outputs = PipelineOutputs(
            results=tuple(results),
            final_artifacts=tuple(final_artifacts), #TODO
            debug_artifacts=tuple(debug_artifacts), #TODO
            metadata=PipelineRunMetadata(
                pipeline_id=self._resolved_pipeline_id(),
                generated_at=datetime.now(timezone.utc),
                execution_metadata=dict(self._execution_metadata),
            ),
            intermediate_frames=intermediate_frames, #TODO
        )
        return self._with_reproducibility_exports(outputs)
        """Execute the full pipeline and return results plus visual artifacts."""
        """
        #VECCHIO PIPELINE SENZA PARALLELISMO
        self._inject_event_bus(self._event_bus)
        ctx = self._context

        buffer = self._run_step("frame_extraction", lambda: ctx.frame_extractor.extract())
        intermediate_store = IntermediateFrameArtifactStore(ctx.intermediate_frame_capture)
        buffer = self._run_step(
            "frame_processing",
            lambda: self._frame_processing_stage.apply(
                buffer,
                ctx.frame_processors,
                intermediate_store=intermediate_store,
            ),
        )
        intermediate_frames = intermediate_store.to_collection()
        buffer, frame_export_artifacts = self._run_frame_exporters(buffer)

        signal = self._run_step("signal_extraction", lambda: ctx.signal_extractor.extract(buffer))

        for i, cleaner in enumerate(ctx.signal_cleaners):
            signal = self._run_step(
                f"signal_cleaning[{i}]",
                lambda c=cleaner: c.clean(signal),  # noqa: B023
            )

        results: list[IData] = []
        for i, analyzer in enumerate(ctx.analyzers):
            data = self._run_step(f"analysis[{i}]", lambda a=analyzer: a.analyze(signal))
            results.append(data)

        final_artifacts = [*frame_export_artifacts, *self._run_visualizers(results)]
        debug_artifacts = self._run_intermediate_frame_visualizers(intermediate_frames)
        outputs = PipelineOutputs(
            results=tuple(results),
            final_artifacts=tuple(final_artifacts),
            debug_artifacts=tuple(debug_artifacts),
            metadata=PipelineRunMetadata(
                pipeline_id=self._resolved_pipeline_id(),
                generated_at=datetime.now(timezone.utc),
                execution_metadata=dict(self._execution_metadata),
            ),
            intermediate_frames=intermediate_frames,
        )
        return self._with_reproducibility_exports(outputs)
        """

    # ── Internals ───────────────────────────────────────────────────────────

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

    def _run_visualizers_streaming(
            self,
            data_buffers: list[DataBuffer],
            artifacts: list[VisualArtifact],
    ) -> list[threading.Thread]:

        bindings = [
            *(VisualizerBinding(v) for v in self._context.visualizers),
            *self._context.visualizer_bindings,
        ]

        threads: list[threading.Thread] = []

        data_buffers_consumer_number = [0] * len(data_buffers)

        #mi annoto il numero di consumatori per ogni data_buffer
        for binding_index, binding in enumerate(bindings):
            target_indexes = self._run_step(
                f"visualisation[{binding_index}].targets",
                lambda b=binding: self._resolve_visualizer_targets(
                    b,
                    len(data_buffers),
                ),
            )

            for result_index in target_indexes:
                data_buffers_consumer_number[result_index] += 1

        for index, data_buffer in enumerate(data_buffers):
            data_buffer._consumers_default = data_buffers_consumer_number[index]


        for binding_index, binding in enumerate(bindings):

            target_indexes = self._run_step(
                f"visualisation[{binding_index}].targets",
                lambda b=binding: self._resolve_visualizer_targets(
                    b,
                    len(data_buffers),
                ),
            )

            for result_index in target_indexes:
                #qui ho usato la formula dell'associazione di Cantor dell'insieme N^2 su N (in pratica
                #associo in modo univoco ogni coppia di naturali (binding_index,result_index) ad un
                #naturale singolo, in modo che l'id di subscribe sia univoco
                buffer = data_buffers[result_index].subscribe(_cantor(binding_index, result_index))

                thread = threading.Thread(
                    target=lambda
                        b=binding,
                        idx=binding_index,
                        ridx=result_index,
                        data=buffer: artifacts.extend(
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
                        )
                    ),
                    daemon=True,
                )

                thread.start()

                threads.append(thread)

        return threads

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
