from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from library.core.pipeline.PipelineErrors import PipelineExecutionError

log = logging.getLogger(__name__)

T = TypeVar("T")


class PipelineStageExecutor:
    """
    Executes one named pipeline stage and normalizes failures.

    Keeping this concern outside ``Pipeline`` makes error handling reusable by
    frame, signal, visualization, and streaming runtime collaborators. Each
    collaborator can stay focused on its own workflow while all stages still
    produce the same diagnostic error shape.

    Stage naming convention
    -----------------------
    Names should be stable and specific enough for logs, UI diagnostics and
    retry decisions. Examples: ``frame_extraction``,
    ``frame_processing[2]``, ``analysis[0]``.
    """

    def run(self, stage: str, operation: Callable[[], T]) -> T:
        """
        Run ``operation`` and wrap raised exceptions with stage context.

        The original exception is preserved as ``PipelineExecutionError.cause``
        so callers can still inspect low-level failures when needed.
        """
        log.debug("Pipeline stage starting: %s", stage)
        try:
            result = operation()
        except Exception as exc:
            log.error("Pipeline stage FAILED: %s - %s", stage, exc, exc_info=True)
            raise PipelineExecutionError(stage, exc) from exc
        log.debug("Pipeline stage completed: %s", stage)
        return result
