from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sef.core.events.Event import Event
from sef.core.pipeline.PipelineErrors import InvalidPipelineTriggerEventError


@dataclass(frozen=True, slots=True)
class PipelineTrigger:
    """
    Typed representation of a pipeline trigger event payload.

    This value is returned by `PipelineEvent.parse()` after validating the
    generic event payload.
    """

    config: Mapping[str, Any]


class PipelineEvent:
    """
    Factory helper for pipeline trigger events.

    The unified event contract is ``Event``. This helper only centralises the
    trigger event type name and construction logic.
    """

    event_type = "pipeline.trigger"

    @staticmethod
    def create(
        config: Mapping[str, Any],
        source: str,
        correlation_id: str | None = None,
    ) -> Event:
        """
        Create a generic event that requests pipeline execution.

        Parameters
        ----------
        config:
            Run config to execute. It must contain its own optional ``id`` and
            ``metadata`` fields.
        source:
            Event producer identifier.
        correlation_id:
            Optional correlation id. Defaults to `pipeline_id`.
        """
        if not isinstance(config, Mapping):
            raise InvalidPipelineTriggerEventError(
                "Pipeline trigger config must be a mapping."
            )
        run_id = _config_id(config)

        return Event(
            event_type=PipelineEvent.event_type,
            source=source,
            correlation_id=correlation_id or run_id,
            payload={"config": dict(config)},
        )

    @staticmethod
    def parse(event: Event) -> PipelineTrigger:
        """Parse and validate a unified Event as a pipeline trigger."""
        if event.event_type != PipelineEvent.event_type:
            raise InvalidPipelineTriggerEventError(
                f"Expected event type '{PipelineEvent.event_type}', got '{event.event_type}'."
            )

        config = event.payload.get("config")
        if config is None:
            raise InvalidPipelineTriggerEventError(
                "Pipeline trigger payload is missing required field 'config'."
            )
        if not isinstance(config, Mapping):
            raise InvalidPipelineTriggerEventError(
                "Pipeline trigger payload field 'config' must be a mapping."
            )

        return PipelineTrigger(
            config=dict(config),
        )

def _config_id(config: Mapping[str, Any]) -> str | None:
    value = config.get("id")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise InvalidPipelineTriggerEventError("Pipeline trigger config field 'id' must be a non-empty string.")
    return value
