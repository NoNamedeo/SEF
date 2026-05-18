from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.pipeline.PipelineContext import PipelineContext

log = logging.getLogger(__name__)


class PipelineEventInjector:
    """
    Injects event dependencies into event-aware pipeline components.

    Event emission is an optional adapter concern. Keeping it isolated avoids
    spreading ``IEventEmitter`` checks across the execution code.
    """

    def inject(
        self,
        *,
        context: PipelineContext,
        event_bus: IEventBus | None,
        metadata: Mapping[str, Any],
    ) -> None:
        """Attach the current event bus and runtime metadata to all emitters."""
        for component in self._components(context):
            if isinstance(component, IEventEmitter):
                component.event_bus = event_bus
                component.event_metadata = dict(metadata)
                log.debug("Injected event context into %s", type(component).__name__)

    @staticmethod
    def _components(context: PipelineContext) -> list[Any]:
        return [
            context.frame_extractor,
            context.signal_extractor,
            *context.frame_processors,
            *context.frame_exporters,
            *context.signal_cleaners,
            *context.analyzers,
            *context.visualizers,
            *(binding.visualizer for binding in context.visualizer_bindings),
            *context.intermediate_frame_visualizers,
        ]
