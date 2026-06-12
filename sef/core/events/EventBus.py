from __future__ import annotations

import logging
import threading
from typing import Callable

from sef.core.events.Event import Event
from sef.core.interfaces.pipeline.IEventBus import IEventBus

log = logging.getLogger(__name__)

EventHandler = Callable[[Event], None]


class EventBus(IEventBus):
    """
    Thread-safe pub/sub event bus implementing IEventBus.

    Two publication paths
    --------------------
    * ``dispatch`` — synchronous; used by pipeline components and
      BranchingCoordinator inside execution threads.
    * ``publish``  — async wrapper around ``dispatch``; for callers
      operating in an asyncio context.

    Handler isolation
    -----------------
    A failing handler is logged and skipped; it never crashes the
    publisher or prevents remaining handlers from running.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handlers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for one event type or wildcard events."""
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a handler if it is registered for the event type."""
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def dispatch(self, event: Event) -> None:
        """
        Dispatch an event synchronously to matching handlers.

        Handler failures are logged and ignored so publication remains isolated.
        """
        event_type = event.event_type

        with self._lock:
            specific = list(self._handlers.get(event_type, []))
            wildcards = list(self._handlers.get(self.WILDCARD, []))

        for handler in (*specific, *wildcards):
            try:
                handler(event)
            except Exception as exc:
                log.warning(
                    "EventBus handler raised (ignored): %s — event: %s (%s)",
                    exc,
                    event_type,
                    event.event_id,
                )

    async def publish(self, event: Event) -> None:
        """Async-compatible wrapper around `dispatch()`."""
        self.dispatch(event)

    def clear(self) -> None:
        """Remove all registered handlers."""
        with self._lock:
            self._handlers.clear()

    def has_subscribers(self, event_type: str) -> bool:
        """Return whether a type-specific or wildcard handler is registered."""
        with self._lock:
            return bool(self._handlers.get(event_type)) or bool(self._handlers.get(self.WILDCARD))
