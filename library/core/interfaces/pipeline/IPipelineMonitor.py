from __future__ import annotations

from abc import ABC, abstractmethod


class IPipelineMonitor(ABC):
    """Tracks active pipelines, supports termination and listing."""

    @abstractmethod
    def register(self, pipeline_id: str) -> None: ...

    @abstractmethod
    def complete(self, pipeline_id: str) -> None: ...

    @abstractmethod
    def terminate(self, pipeline_id: str) -> None: ...

    @abstractmethod
    def active_ids(self) -> list[str]: ...
