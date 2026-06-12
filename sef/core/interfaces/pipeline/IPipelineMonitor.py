from __future__ import annotations

from abc import ABC, abstractmethod

from sef.core.pipeline.PipelineRunSnapshot import PipelineRunSnapshot


class IPipelineMonitor(ABC):
    """
    Port for observable pipeline execution state.

    Monitors are read by UIs, APIs, and operators. Implementations should keep
    state transitions deterministic and should not expose mutable internal
    storage through snapshots.
    """

    @abstractmethod
    def register(self, pipeline_id: str) -> None:
        """Record a newly queued pipeline run."""
        ...

    @abstractmethod
    def mark_running(self, pipeline_id: str, attempt: int) -> None:
        """Mark a run attempt as actively executing."""
        ...

    @abstractmethod
    def complete(self, pipeline_id: str) -> None:
        """Mark a run as successfully completed."""
        ...

    @abstractmethod
    def fail(self, pipeline_id: str, error: Exception | str, attempt: int) -> None:
        """Mark a run as failed with an observable error message."""
        ...

    @abstractmethod
    def terminate(self, pipeline_id: str) -> None:
        """Mark a run as cancelled or terminated."""
        ...

    @abstractmethod
    def active_ids(self) -> list[str]:
        """Return a snapshot of queued or running pipeline ids."""
        ...

    @abstractmethod
    def snapshot(self, pipeline_id: str) -> PipelineRunSnapshot | None:
        """Return the latest snapshot for a run id, if known."""
        ...

    @abstractmethod
    def snapshots(self) -> list[PipelineRunSnapshot]:
        """Return latest snapshots for all tracked runs."""
        ...
