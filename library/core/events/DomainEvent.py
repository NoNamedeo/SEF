from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DomainEvent:
    """
    Immutable domain event emitted by pipeline components.

    Design rationale
    ----------------
    Domain events are distinct from lifecycle events (BEFORE_RUN, AFTER_RUN, etc.).
    They represent **business-level signals** originating from pipeline components
    during execution — for example a SignalExtractor detecting that tracking was
    lost, or an Analyzer spotting an anomaly in the data.

    The Orchestrator can subscribe to these events and use IBranchingRule
    strategies to decide whether to spawn secondary pipelines automatically.

    frozen=True guarantees thread-safe sharing between the publisher thread
    and any consumer thread that picks up the event.

    Attributes
    ----------
    event_type : str
        A short, namespaced identifier for the kind of event
        (e.g. ``"tracking_lost"``, ``"object_detected"``).
    source : str
        Human-readable name of the component that emitted the event
        (e.g. ``"OpenCVBufferedSignalExtractor"``).
    payload : dict[str, Any]
        Arbitrary data attached to the event.  The schema is defined
        by the emitting component; consumers must handle unknown keys
        gracefully.
    timestamp : float
        Wall-clock time (``time.time()``) when the event was created.
        Automatically set at construction time.
    """

    event_type: str
    source:     str
    payload:    dict[str, Any] = field(default_factory=dict)
    timestamp:  float          = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            f"DomainEvent(type={self.event_type!r}, "
            f"source={self.source!r}, "
            f"payload_keys={list(self.payload.keys())}, "
            f"timestamp={self.timestamp:.3f})"
        )
