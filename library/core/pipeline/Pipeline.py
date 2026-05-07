from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from library.core.artifacts.IntermediateFrameArtifacts import IntermediateFrameArtifactCollection
from library.core.interfaces.IData import IData
from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.pipeline.FrameCleaningStage import FrameCleaningStage
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.VisualizerBinding import VisualizerBinding
from library.core.visualization.PipelineOutputs import PipelineOutputs
from library.core.visualization.PipelineRunMetadata import PipelineRunMetadata
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext

log = logging.getLogger(__name__)


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
    2. Frame cleaning     (frame_cleaners   → cleaned buffer)   [optional]
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
        self._frame_cleaning_stage = FrameCleaningStage()

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> PipelineOutputs:
        """Execute the full pipeline and return results plus visual artifacts."""
        self._inject_event_bus(self._event_bus)
        ctx = self._context

        buffer = self._run_step("frame_extraction", lambda: ctx.frame_extractor.extract())
        intermediate_store = IntermediateFrameArtifactStore(ctx.intermediate_frame_capture)
        buffer = self._run_step(
            "frame_cleaning",
            lambda: self._frame_cleaning_stage.apply(
                buffer,
                ctx.frame_cleaners,
                intermediate_store=intermediate_store,
            ),
        )
        intermediate_frames = intermediate_store.to_collection()

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

        artifacts = self._run_visualizers(results)
        artifacts.extend(self._run_intermediate_frame_visualizers(intermediate_frames))
        outputs = PipelineOutputs(
            results=tuple(results),
            artifacts=tuple(artifacts),
            metadata=PipelineRunMetadata(
                pipeline_id=self._resolved_pipeline_id(),
                generated_at=datetime.now(timezone.utc),
                execution_metadata=dict(self._execution_metadata),
            ),
            intermediate_frames=intermediate_frames,
        )
        return self._with_reproducibility_exports(outputs)

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
            *self._context.frame_cleaners,
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
            artifacts=outputs.artifacts,
            metadata=PipelineRunMetadata(
                pipeline_id=outputs.metadata.pipeline_id,
                generated_at=outputs.metadata.generated_at,
                execution_metadata=outputs.metadata.execution_metadata,
                reproducibility=reproducibility,
            ),
            intermediate_frames=outputs.intermediate_frames,
        )

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
