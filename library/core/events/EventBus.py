from __future__ import annotations

import logging
import threading
from typing import Callable

from library.core.events.DomainEvent import DomainEvent

log = logging.getLogger(__name__)

# Public type alias for handler signatures
DomainEventHandler = Callable[[DomainEvent], None]


class EventBus:
    """
    Thread-safe publish/subscribe bus for domain events.

    Design rationale
    ----------------
    This EventBus is **separate** from the PipelineOrchestrator's lifecycle
    event system (BEFORE_RUN, AFTER_RUN …).  It carries domain-level
    signals — events emitted *during* pipeline execution by components
    such as SignalExtractors or Analyzers.

    The Orchestrator subscribes to this bus so it can evaluate
    IBranchingRule strategies and spawn secondary pipelines automatically.

    Thread safety
    -------------
    All mutations (subscribe, unsubscribe, clear) and reads (publish) are
    protected by a ``threading.Lock``.  This is necessary because:

    * Pipeline components emit events from the pipeline thread.
    * The Orchestrator may run secondary pipelines on a ThreadPoolExecutor,
      which may themselves emit events concurrently.

    Handler isolation
    -----------------
    If a handler raises an exception it is logged and silently ignored —
    exactly the same contract used by PipelineOrchestrator's lifecycle
    handlers.  One faulty subscriber must never crash the publisher or
    prevent other subscribers from being notified.

    Example
    -------
    >>> bus = EventBus()
    >>> bus.subscribe("tracking_lost", lambda e: print(e))
    >>> bus.publish(DomainEvent("tracking_lost", "MySE", {"frame": 42}))
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handlers: dict[str, list[DomainEventHandler]] = {}
        self._wildcard_handlers: list[DomainEventHandler] = []

    # ── Subscribe ────────────────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: str,
        handler: DomainEventHandler,
    ) -> None:
        """
        Register *handler* to be called whenever an event with
        *event_type* is published.

        Parameters
        ----------
        event_type:
            The DomainEvent.event_type to listen for.
        handler:
            Callable receiving a single DomainEvent argument.
        """
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: DomainEventHandler) -> None:
        """
        Register a wildcard handler called for **every** published event,
        regardless of event_type.

        Useful for logging, metrics collection, or debugging.
        """
        with self._lock:
            self._wildcard_handlers.append(handler)

    # ── Unsubscribe ──────────────────────────────────────────────────────────

    def unsubscribe(
        self,
        event_type: str,
        handler: DomainEventHandler,
    ) -> None:
        """Remove a previously registered handler (no-op if not found)."""
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def unsubscribe_all(self, handler: DomainEventHandler) -> None:
        """Remove a previously registered wildcard handler (no-op if not found)."""
        with self._lock:
            try:
                self._wildcard_handlers.remove(handler)
            except ValueError:
                pass

    # ── Publish ──────────────────────────────────────────────────────────────

    def publish(self, event: DomainEvent) -> None:
        """
        Dispatch *event* to all matching handlers synchronously.

        Handlers are called in registration order.  If a handler raises,
        the exception is logged and the remaining handlers still execute.
        """
        with self._lock:
            specific = list(self._handlers.get(event.event_type, []))
            wildcards = list(self._wildcard_handlers)

        for handler in (*specific, *wildcards):
            try:
                handler(event)
            except Exception as exc:
                log.warning(
                    "EventBus handler raised an exception (ignored): %s — event: %s",
                    exc,
                    event.event_type,
                )

    # ── Management ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all handlers (specific and wildcard)."""
        with self._lock:
            self._handlers.clear()
            self._wildcard_handlers.clear()

    def has_subscribers(self, event_type: str) -> bool:
        """Return True if at least one handler is registered for *event_type*."""
        with self._lock:
            return bool(self._handlers.get(event_type)) or bool(self._wildcard_handlers)
