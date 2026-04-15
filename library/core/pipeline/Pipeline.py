from __future__ import annotations

import logging
from typing import Any, Callable

from library.core.interfaces.IData import IData
from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.pipeline.FrameCleaningStage import FrameCleaningStage
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.VisualizerBinding import VisualizerBinding

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
    ) -> None:
        self._context = context
        self._event_bus = event_bus
        self._pipeline_id = pipeline_id
        self._frame_cleaning_stage = FrameCleaningStage()

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> list[IData]:
        """Execute the full pipeline and return one IData per analyzer."""
        self._inject_event_bus(self._event_bus)
        ctx = self._context

        buffer = self._run_step("frame_extraction", lambda: ctx.frame_extractor.extract())
        buffer = self._run_step(
            "frame_cleaning",
            lambda: self._frame_cleaning_stage.apply(buffer, ctx.frame_cleaners),
        )

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

        self._run_visualizers(results)

        return results

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
        if self._pipeline_id is None:
            return {}
        return {"pipeline_id": self._pipeline_id}

    def _run_visualizers(self, results: list[IData]) -> None:
        bindings = [
            *(VisualizerBinding(visualizer) for visualizer in self._context.visualizers),
            *self._context.visualizer_bindings,
        ]

        for binding_index, binding in enumerate(bindings):
            target_indexes = self._run_step(
                f"visualisation[{binding_index}].targets",
                lambda b=binding: self._resolve_visualizer_targets(b, len(results)),
            )
            for result_index in target_indexes:
                data = results[result_index]
                self._run_step(
                    f"visualisation[{binding_index}][{result_index}]",
                    lambda v=binding.visualizer, d=data: v.visualize(d),
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
            raise ValueError(
                f"Visualizer target index out of range: {invalid}; "
                f"available result indexes: 0..{result_count - 1}."
            )
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
