from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.pipeline.Pipeline import Pipeline
from library.core.pipeline.PipelineContext import PipelineContext


class IPipelineFactory(ABC):
    """Creates executable Pipeline instances from validated contexts."""

    @abstractmethod
    def create(
        self,
        context: PipelineContext,
        event_bus: IEventBus | None = None,
        pipeline_id: str | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> Pipeline: ...
