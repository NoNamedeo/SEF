from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, Callable

from library.core.abstractions.IData import IData
from library.core.pipeline.PipelineContext import PipelineContext

if TYPE_CHECKING:
    from library.core.pipeline.Pipeline import Pipeline

log = logging.getLogger(__name__)


# ── Lifecycle event types ───────────────────────────────────────────────────


class PipelineEvent(StrEnum):
    """
    Well-known lifecycle events emitted during pipeline execution.

    Using StrEnum keeps event names both type-safe (IDE autocompletion,
    mypy checks) and human-readable in logs/serialisation.
    """

    BEFORE_RUN = auto()  # emitted before Pipeline.run() is called
    AFTER_RUN = auto()  # emitted after a successful run
    ON_ERROR = auto()  # emitted when PipelineExecutionError is caught
    ON_RETRY = auto()  # emitted before each retry attempt


@dataclass
class PipelineEventPayload:
    """
    Data bag passed to every lifecycle event handler.

    Fields
    ------
    event:
        Which lifecycle event triggered this payload.
    context:
        The PipelineContext of the pipeline that emitted.
    results:
        Analysis results (populated on AFTER_RUN).
    error:
        The exception (populated on ON_ERROR).
    attempt:
        The 1-based attempt number (populated on ON_ERROR / ON_RETRY).
    pipeline:
        The Pipeline instance (informational, optional).
    pipeline_id:
        Human-readable identifier — ``"primary"`` for the main pipeline,
        ``"secondary-<n>"`` for auto-spawned ones, or any custom string.
        Allows handlers to distinguish which pipeline emitted the event.
    """

    event: PipelineEvent
    context: PipelineContext
    results: list[IData] = field(default_factory=list)
    error: Exception | None = field(default=None)
    attempt: int = field(default=1)
    pipeline: Any = field(default=None)  # Pipeline instance (optional)
    pipeline_id: str = field(default="primary")


LifecycleEventHandler = Callable[[PipelineEventPayload], None]

# Backward-compatible alias (was ``EventHandler`` in the original Orchestrator module)
EventHandler = LifecycleEventHandler


# ── Bus ─────────────────────────────────────────────────────────────────────


class PipelineLifecycleBus:
    """
    Thread-safe pub/sub for pipeline lifecycle events.

    Design rationale
    ----------------
    Lifecycle event handling used to live inside ``PipelineOrchestrator``,
    making it non-shareable: secondary pipelines (spawned on worker threads)
    could never emit lifecycle events visible to the primary's subscribers.

    By extracting the handler registry into a standalone, **injectable** bus:

    * The same bus can be passed to primary AND secondary orchestrators,
      so all lifecycle events (BEFORE_RUN, AFTER_RUN, ON_ERROR, ON_RETRY)
      arrive at the same set of handlers.
    * Each orchestrator remains thin — it just calls ``bus.emit(payload)``.
    * Thread safety is built in: secondary pipelines emit from worker
      threads, while subscribe/unsubscribe may happen on the main thread.

    Example
    -------
    >>> bus = PipelineLifecycleBus()
    >>> bus.subscribe(PipelineEvent.AFTER_RUN, lambda p: print(p.results))
    >>> # pass `bus` to both primary and secondary orchestrators
    """

    def __init__(self) -> None:
        self._handlers: dict[PipelineEvent, list[LifecycleEventHandler]] = {
            e: [] for e in PipelineEvent
        }
        self._lock = threading.Lock()

    # ── Subscribe / Unsubscribe ─────────────────────────────────────────────

    def subscribe(self, event: PipelineEvent, handler: LifecycleEventHandler) -> None:
        """Register *handler* to be called when *event* is emitted."""
        with self._lock:
            self._handlers[event].append(handler)

    def unsubscribe(self, event: PipelineEvent, handler: LifecycleEventHandler) -> None:
        """Remove a previously registered handler (no-op if not found)."""
        with self._lock:
            try:
                self._handlers[event].remove(handler)
            except ValueError:
                pass

    # ── Emit ────────────────────────────────────────────────────────────────

    def emit(self, payload: PipelineEventPayload) -> None:
        """
        Dispatch *payload* to all handlers registered for ``payload.event``.

        Handlers are called in registration order.  If a handler raises,
        the exception is logged and silently ignored — one faulty handler
        must never crash the pipeline or prevent other handlers from running.
        """
        with self._lock:
            handlers = list(self._handlers[payload.event])

        for handler in handlers:
            try:
                handler(payload)
            except Exception as exc:
                log.warning(
                    "Lifecycle handler raised an exception (ignored): %s",
                    exc,
                )
