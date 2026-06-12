from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Event:
    """
    Canonical immutable event exchanged across the core.

    Every event in the system shares the same contract regardless of whether
    it originates from a pipeline component, the orchestrator, or lifecycle
    instrumentation. Specialised modules may expose helper constructors and
    event type constants, but the event object itself is always this class.
    """

    event_type: str
    source: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("Event.event_type must be a non-empty string.")
        if not self.source:
            raise ValueError("Event.source must be a non-empty string.")

        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def get(self, key: str, default: Any = None) -> Any:
        """Return a payload value or *default* when the key is absent."""
        return self.payload.get(key, default)

    def require(self, key: str) -> Any:
        """
        Return the payload value for *key* or raise KeyError.

        This keeps event consumers explicit about which payload entries are
        mandatory for a given event type.
        """
        if key not in self.payload:
            raise KeyError(
                f"Missing payload key '{key}' in event '{self.event_type}' "
                f"from '{self.source}'."
            )
        return self.payload[key]

