from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from library.core.abstractions.IData import IData
from library.core.pipeline.PipelineContext import PipelineContext

if TYPE_CHECKING:
    pass

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

    Raises
    ------
    PipelineExecutionError
        Wraps any exception raised by a pipeline step, enriching it with
        the name of the failing stage so callers can act accordingly.
    """

    def __init__(self, context: PipelineContext) -> None:
        self._context = context

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> list[IData]:
        """Execute the full pipeline and return one IData per analyzer."""
        ctx = self._context

        buffer = self._run_step("frame_extraction",
                                lambda: ctx.frame_extractor.extract(ctx.frame_cleaners))

        signal = self._run_step("signal_extraction",
                                lambda: ctx.signal_extractor.extract(buffer))

        for i, cleaner in enumerate(ctx.signal_cleaners):
            signal = self._run_step(f"signal_cleaning[{i}]",
                                    lambda c=cleaner: c.clean(signal))

        results: list[IData] = []
        for i, analyzer in enumerate(ctx.analyzers):
            data = self._run_step(f"analysis[{i}]",
                                  lambda a=analyzer: a.analyze(signal))
            results.append(data)

        for i, visualizer in enumerate(ctx.visualizers):
            for j, data in enumerate(results):
                self._run_step(f"visualisation[{i}][{j}]",
                               lambda v=visualizer, d=data: v.visualize(d))

        return results

    # ── Internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _run_step(stage: str, fn):
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