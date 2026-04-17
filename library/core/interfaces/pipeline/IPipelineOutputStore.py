from __future__ import annotations

from abc import ABC, abstractmethod

from library.core.visualization.PipelineOutputs import PipelineOutputs


class IPipelineOutputStore(ABC):
    """Persistence port for completed pipeline outputs."""

    @abstractmethod
    def save(self, pipeline_id: str, outputs: PipelineOutputs) -> None: ...

    @abstractmethod
    def get(self, pipeline_id: str) -> PipelineOutputs | None: ...

    @abstractmethod
    def delete(self, pipeline_id: str) -> None: ...
