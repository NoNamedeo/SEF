from __future__ import annotations

from enum import StrEnum

from library.core.events.Event import Event


class PipelineLifecycleEvent(StrEnum):
    BEFORE_RUN = "pipeline.before_run"
    AFTER_RUN = "pipeline.after_run"
    ON_ERROR = "pipeline.error"
    ON_RETRY = "pipeline.retry"
    CANCELLED = "pipeline.cancelled"
    REJECTED = "pipeline.rejected"
    SUBMIT_FAILED = "pipeline.submit_failed"


def create_pipeline_lifecycle_event(
    event: PipelineLifecycleEvent,
    pipeline_id: str,
    source: str,
    result_count: int | None = None,
    artifact_count: int | None = None,
    error: Exception | None = None,
    attempt: int = 1,
    correlation_id: str | None = None,
) -> Event:
    """Create a lifecycle event using the unified Event contract."""
    return Event(
        event_type=str(event),
        source=source,
        correlation_id=correlation_id or pipeline_id,
        payload={
            "pipeline_id": pipeline_id,
            "result_count": result_count,
            "artifact_count": artifact_count,
            "error": error,
            "attempt": attempt,
        },
    )


__all__ = ["PipelineLifecycleEvent", "create_pipeline_lifecycle_event"]
