from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from library.core.events.Event import Event
from library.core.pipeline.PipelineErrors import InvalidPipelineTriggerEventError
from library.core.pipeline.PipelineContext import PipelineContext


@dataclass(frozen=True, slots=True)
class PipelineTrigger:
    """Typed representation of a pipeline trigger event payload."""

    pipeline_id: str
    context: PipelineContext


class PipelineEvent:
    """
    Factory helper for pipeline trigger events.

    The unified event contract is ``Event``. This helper only centralises the
    trigger event type name and construction logic.
    """

    event_type = "pipeline.trigger"

    @staticmethod
    def create(
        pipeline_id: str,
        context: PipelineContext,
        source: str,
        correlation_id: str | None = None,
    ) -> Event:
        return Event(
            event_type=PipelineEvent.event_type,
            source=source,
            correlation_id=correlation_id or pipeline_id,
            payload={
                "pipeline_id": pipeline_id,
                "context": context,
            },
        )

    @staticmethod
    def parse(event: Event) -> PipelineTrigger:
        """Parse and validate a unified Event as a pipeline trigger."""
        if event.event_type != PipelineEvent.event_type:
            raise InvalidPipelineTriggerEventError(
                f"Expected event type '{PipelineEvent.event_type}', got '{event.event_type}'."
            )

        pipeline_id = PipelineEvent._require_string(event, "pipeline_id")
        try:
            context = event.require("context")
        except KeyError as exc:
            raise InvalidPipelineTriggerEventError(
                "Pipeline trigger payload is missing required field 'context'."
            ) from exc
        if not isinstance(context, PipelineContext):
            raise InvalidPipelineTriggerEventError(
                "Pipeline trigger payload field 'context' must be a PipelineContext."
            )

        return PipelineTrigger(pipeline_id=pipeline_id, context=context)

    @staticmethod
    def _require_string(event: Event, key: str) -> str:
        try:
            value: Any = event.require(key)
        except KeyError as exc:
            raise InvalidPipelineTriggerEventError(
                f"Pipeline trigger payload is missing required field '{key}'."
            ) from exc
        if not isinstance(value, str) or not value:
            raise InvalidPipelineTriggerEventError(
                f"Pipeline trigger payload field '{key}' must be a non-empty string."
            )
        return value
