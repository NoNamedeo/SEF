from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from library.core.events.Event import Event
from library.core.interfaces.pipeline.IEventBus import IEventBus

log = logging.getLogger(__name__)


class IEventEmitter:
    """
    Mixin that gives any pipeline component the ability to emit domain events.

    Design rationale
    ----------------
    IEventEmitter is intentionally a **concrete mixin**, NOT an ABC.
    This means:

    * Existing components (ISignalExtractor, IAnalyzer …) do NOT need to
      change — they only gain event emission capability if their concrete
      subclass explicitly inherits from this mixin.
    * The ``emit()`` method is already implemented: subclasses just call
      ``self.emit("tracking_lost", {"frame": 42})`` and the rest is handled.
    * If no EventBus has been injected (the default), ``emit()`` is a
      silent no-op, preserving backward compatibility.

    The EventBus is injected by the Pipeline before execution via the
    ``event_bus`` property setter.  Components never create an EventBus
    themselves — this follows the Dependency Inversion Principle.

    Usage in a concrete component
    -----------------------------
    >>> class MySignalExtractor(ISignalExtractor, IEventEmitter):
    ...     def extract(self, buffer: FrameBuffer) -> ISignal:
    ...         for frame in buffer:
    ...             if tracking_lost(frame):
    ...                 self.emit("tracking_lost", {"frame_index": frame.index})
    ...         return Signal(samples)
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure the mixin initialiser is called even in diamond MRO."""
        super().__init_subclass__(**kwargs)

    # ── EventBus injection ───────────────────────────────────────────────────

    @property
    def event_bus(self) -> IEventBus | None:
        """Return the currently injected EventBus, or None."""
        return getattr(self, "_event_bus", None)

    @event_bus.setter
    def event_bus(self, bus: IEventBus | None) -> None:
        """
        Inject an EventBus.

        Called by Pipeline._inject_event_bus() before execution.
        """
        self._event_bus = bus

    @property
    def event_metadata(self) -> Mapping[str, Any]:
        """Return metadata automatically attached to emitted events."""
        return getattr(self, "_event_metadata", {})

    @event_metadata.setter
    def event_metadata(self, metadata: Mapping[str, Any]) -> None:
        """
        Inject execution metadata.

        Pipeline uses this to propagate execution-scoped values, such as the
        current pipeline id, without coupling components to the orchestrator.
        """
        self._event_metadata = dict(metadata)

    # ── Emit convenience ─────────────────────────────────────────────────────

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """
        Publish a domain event on the injected EventBus.

        If no EventBus has been injected this is a **silent no-op** —
        components can always call ``self.emit()`` without checking
        whether they are running inside an event-aware orchestrator.

        Parameters
        ----------
        event_type:
            Short identifier for the event (e.g. ``"tracking_lost"``).
        payload:
            Arbitrary data attached to the event.
        """
        bus = self.event_bus
        if bus is None:
            return

        event_payload = {
            **(payload or {}),
            **self.event_metadata,
        }
        event = Event(
            event_type=event_type,
            source=type(self).__name__,
            payload=event_payload,
        )
        log.debug("IEventEmitter.emit: %s", event)
        bus.dispatch(event)
