from __future__ import annotations

from abc import ABC, abstractmethod

from library.core.pipeline.PipelineRunSnapshot import PipelineRunSnapshot


class IPipelineMonitor(ABC):
    """Tracks observable pipeline execution state."""

    @abstractmethod
    def register(self, pipeline_id: str) -> None: ...

    @abstractmethod
    def mark_running(self, pipeline_id: str, attempt: int) -> None: ...

    @abstractmethod
    def complete(self, pipeline_id: str) -> None: ...

    @abstractmethod
    def fail(self, pipeline_id: str, error: Exception | str, attempt: int) -> None: ...

    @abstractmethod
    def terminate(self, pipeline_id: str) -> None: ...

    @abstractmethod
    def active_ids(self) -> list[str]: ...

    @abstractmethod
    def snapshot(self, pipeline_id: str) -> PipelineRunSnapshot | None: ...

    @abstractmethod
    def snapshots(self) -> list[PipelineRunSnapshot]: ...
