from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.pipeline.Pipeline import Pipeline
from library.core.pipeline.PipelineContext import PipelineContext


class IPipelineFactory(ABC):
    """
    Factory port that creates executable `Pipeline` instances.

    Applications can replace this port to inject custom execution policy,
    event buses, metadata propagation, or pipeline subclasses without changing
    the orchestrator.
    """

    @abstractmethod
    def create(
        self,
        context: PipelineContext,
        event_bus: IEventBus | None = None,
        pipeline_id: str | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> Pipeline:
        """
        Create a pipeline facade for a validated context.

        Returns
        -------
        Pipeline
            Executable pipeline instance ready for a runner.
        """
        ...
