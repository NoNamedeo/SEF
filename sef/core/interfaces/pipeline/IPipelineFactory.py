from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from sef.core.interfaces.pipeline.IEventBus import IEventBus
from sef.core.pipeline.Pipeline import Pipeline
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineRunOptions import PipelineRunOptions


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
        run_options: PipelineRunOptions | None = None,
    ) -> Pipeline:
        """
        Create a pipeline facade for a validated context.

        Returns
        -------
        Pipeline
            Executable pipeline instance ready for a runner.

        Notes
        -----
        Implementations should forward ``run_options`` without eagerly
        constructing execution plans or reproducibility exports.
        """
        ...
