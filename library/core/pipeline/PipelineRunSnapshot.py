from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PipelineRunState(StrEnum):
    """Observable lifecycle states for a pipeline run."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PipelineRunSnapshot:
    """
    Immutable snapshot of a pipeline run.

    Timestamps use ``time.time()`` seconds to stay consistent with the existing
    event timestamp model.
    """

    pipeline_id: str
    state: PipelineRunState
    attempt: int = 0
    error: str | None = None
    submitted_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None

