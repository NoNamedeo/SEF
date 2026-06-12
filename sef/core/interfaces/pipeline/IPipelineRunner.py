from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future

from library.core.pipeline.Pipeline import Pipeline
from library.core.pipeline.PipelineRunSnapshot import PipelineRunSnapshot
from library.core.visualization.PipelineOutputs import PipelineOutputs


class IPipelineRunner(ABC):
    """
    Executes Pipeline instances synchronously or asynchronously.

    Implementations own execution concerns such as active-run tracking,
    retry, lifecycle events, and monitor registration/completion.
    """

    @abstractmethod
    def run(self, pipeline_id: str, pipeline: Pipeline) -> PipelineOutputs:
        """Execute a pipeline synchronously and return completed outputs."""
        ...

    @abstractmethod
    def submit(self, pipeline_id: str, pipeline: Pipeline) -> Future[PipelineOutputs]:
        """Submit a pipeline for asynchronous execution."""
        ...

    @abstractmethod
    def cancel(self, pipeline_id: str) -> bool:
        """Cancel queued work for a run id when supported."""
        ...

    @abstractmethod
    def active_ids(self) -> list[str]:
        """Return a snapshot of active run ids."""
        ...

    @abstractmethod
    def snapshot(self, pipeline_id: str) -> PipelineRunSnapshot | None:
        """Return the latest snapshot for a run id, if known."""
        ...

    @abstractmethod
    def snapshots(self) -> list[PipelineRunSnapshot]:
        """Return latest snapshots for all tracked runs."""
        ...

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Release runner resources and optionally wait for active work."""
        ...
