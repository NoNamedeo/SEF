from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future

from library.core.interfaces.IData import IData
from library.core.pipeline.Pipeline import Pipeline
from library.core.pipeline.PipelineRunSnapshot import PipelineRunSnapshot


class IPipelineRunner(ABC):
    """
    Executes Pipeline instances synchronously or asynchronously.

    Implementations own execution concerns such as active-run tracking,
    retry, lifecycle events, and monitor registration/completion.
    """

    @abstractmethod
    def run(self, pipeline_id: str, pipeline: Pipeline) -> list[IData]: ...

    @abstractmethod
    def submit(self, pipeline_id: str, pipeline: Pipeline) -> Future[list[IData]]: ...

    @abstractmethod
    def cancel(self, pipeline_id: str) -> bool: ...

    @abstractmethod
    def active_ids(self) -> list[str]: ...

    @abstractmethod
    def snapshot(self, pipeline_id: str) -> PipelineRunSnapshot | None: ...

    @abstractmethod
    def snapshots(self) -> list[PipelineRunSnapshot]: ...

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None: ...
