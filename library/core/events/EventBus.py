from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from library.core.interfaces.pipeline.IEventBus import IEventBus

from library.core.events.DomainEvent import DomainEvent

log = logging.getLogger(__name__)

EventHandler = Callable[[Any], None]
DomainEventHandler = Callable[[DomainEvent], None]


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
        self._wildcard_handlers: list[EventHandler] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        with self._lock:
            self._wildcard_handlers.append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def unsubscribe_all(self, handler: EventHandler) -> None:
        with self._lock:
            try:
                self._wildcard_handlers.remove(handler)
            except ValueError:
                pass

    def dispatch(self, event: Any) -> None:
        if not hasattr(event, "event_type"):
            raise TypeError("Event must have 'event_type'")

        event_type: str = event.event_type

        with self._lock:
            specific = list(self._handlers.get(event_type, []))
            wildcards = list(self._wildcard_handlers)

        for handler in (*specific, *wildcards):
            try:
                handler(event)
            except Exception as exc:
                log.warning(
                    "EventBus handler raised (ignored): %s — event: %s",
                    exc,
                    event_type,
                )

    async def publish(self, event: Any) -> None:
        self.dispatch(event)

    def clear(self) -> None:
        with self._lock:
            self._handlers.clear()
            self._wildcard_handlers.clear()

    def has_subscribers(self, event_type: str) -> bool:
        with self._lock:
            return bool(self._handlers.get(event_type)) or bool(self._wildcard_handlers)
