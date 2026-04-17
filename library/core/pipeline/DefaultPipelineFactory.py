from __future__ import annotations

from typing import Any, Mapping

from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.interfaces.pipeline.IPipelineFactory import IPipelineFactory
from library.core.pipeline.Pipeline import Pipeline
from library.core.pipeline.PipelineContext import PipelineContext


class DefaultPipelineFactory(IPipelineFactory):
    """Default factory that keeps Pipeline construction outside the orchestrator."""

    def create(
        self,
        context: PipelineContext,
        event_bus: IEventBus | None = None,
        pipeline_id: str | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> Pipeline:
        return Pipeline(
            context,
            event_bus=event_bus,
            pipeline_id=pipeline_id,
            execution_metadata=execution_metadata,
        )
