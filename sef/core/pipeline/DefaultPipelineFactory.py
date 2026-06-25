from __future__ import annotations

from typing import Any, Mapping

from sef.core.interfaces.pipeline.IEventBus import IEventBus
from sef.core.interfaces.pipeline.IPipelineFactory import IPipelineFactory
from sef.core.pipeline.Pipeline import Pipeline
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineExecutionPolicy import PipelineExecutionPolicy
from sef.core.pipeline.PipelineRunOptions import PipelineRunOptions


class DefaultPipelineFactory(IPipelineFactory):
    """Default factory that keeps Pipeline construction outside the orchestrator."""

    def __init__(self, execution_policy: PipelineExecutionPolicy | None = None) -> None:
        self._execution_policy = execution_policy

    def create(
        self,
        context: PipelineContext,
        event_bus: IEventBus | None = None,
        pipeline_id: str | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
        run_options: PipelineRunOptions | None = None,
    ) -> Pipeline:
        return Pipeline(
            context,
            event_bus=event_bus,
            pipeline_id=pipeline_id,
            execution_metadata=execution_metadata,
            execution_policy=self._execution_policy,
            run_options=run_options,
        )
