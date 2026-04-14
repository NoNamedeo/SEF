from __future__ import annotations

from abc import ABC, abstractmethod

from library.core.pipeline.Pipeline import Pipeline


class IPipelineRunner(ABC):
    """Submits and executes a Pipeline asynchronously."""

    @abstractmethod
    def submit(self, pipeline_id: str, pipeline: Pipeline) -> None: ...

    @abstractmethod
    def cancel(self, pipeline_id: str) -> None: ...
