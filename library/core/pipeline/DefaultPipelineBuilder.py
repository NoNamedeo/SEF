from __future__ import annotations

# from library.core.artifacts.PipelineEvent import PipelineEvent
from library.core.events.EventBus import EventBus
from library.core.interfaces.pipeline.IPipelineBuilder import IPipelineBuilder
from library.core.pipeline.Pipeline import Pipeline
from library.core.validators.pipeline.PipelineContextValidator import PipelineContextValidator


class DefaultPipelineBuilder(IPipelineBuilder):
    """
    Builds a Pipeline directly from the context carried in a PipelineEvent.

    An optional domain EventBus is injected into every IEventEmitter
    component found in the context, enabling domain-event emission during
    pipeline execution.
    """

    def __init__(self, domain_bus: EventBus | None = None) -> None:
        self._domain_bus = domain_bus

    def build(self, event) -> Pipeline:
        PipelineContextValidator.validate(event.context)
        return Pipeline(event.context, event_bus=self._domain_bus)
