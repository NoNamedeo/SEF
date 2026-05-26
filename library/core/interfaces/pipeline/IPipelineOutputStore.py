from __future__ import annotations

from abc import ABC, abstractmethod

from library.core.visualization.PipelineOutputs import PipelineOutputs


class IPipelineOutputStore(ABC):
    """
    Persistence port for completed pipeline outputs.

    Stores may be in-memory, filesystem-backed, database-backed, or service
    backed. Implementations should treat `PipelineOutputs` as immutable values.
    """

    @abstractmethod
    def save(self, pipeline_id: str, outputs: PipelineOutputs) -> None:
        """Persist outputs for a completed run id."""
        ...

    @abstractmethod
    def get(self, pipeline_id: str) -> PipelineOutputs | None:
        """Return stored outputs for a run id, if available."""
        ...

    @abstractmethod
    def delete(self, pipeline_id: str) -> None:
        """Remove stored outputs for a run id."""
        ...
