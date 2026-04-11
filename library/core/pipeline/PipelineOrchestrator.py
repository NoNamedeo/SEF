from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Callable

from library.core.abstractions.IRetryPolicy import IRetryPolicy
from library.core.pipeline.Pipeline import Pipeline, PipelineExecutionError
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.abstractions.IData import IData
from library.retry_policies.NoRetryPolicy import NoRetryPolicy

log = logging.getLogger(__name__)


# ── Event system ─────────────────────────────────────────────────────────────


class PipelineEvent(StrEnum):
    """
    Well-known lifecycle events emitted by the orchestrator.

    Using StrEnum keeps event names both type-safe (IDE autocompletion,
    mypy checks) and human-readable in logs/serialisation.
    """

    BEFORE_RUN = auto()  # emitted before Pipeline.run() is called
    AFTER_RUN = auto()  # emitted after a successful run
    ON_ERROR = auto()  # emitted when PipelineExecutionError is caught
    ON_RETRY = auto()  # emitted before each retry attempt


@dataclass
class PipelineEventPayload:
    """Data bag passed to every event handler."""

    event: PipelineEvent
    pipeline: Pipeline
    context: PipelineContext
    results: list[IData] = field(default_factory=list)
    error: Exception | None = field(default=None)
    attempt: int = field(default=1)


EventHandler = Callable[[PipelineEventPayload], None]


# ── Orchestrator ──────────────────────────────────────────────────────────────


class PipelineOrchestrator:
    """
    Brain of the pipeline system.

    Design rationale
    ----------------
    The orchestrator owns ALL decisions that Pipeline must not make:

      • which context to use (construction / selection logic)
      • retry policy on transient failures  ← pluggable via IRetryPolicy
      • lifecycle event dispatch (hooks for logging, metrics, UI updates…)
      • conditional branching (run a secondary pipeline based on results)
      • future async scheduling

    This strict separation means Pipeline stays a 'dumb executor' while
    all system-level intelligence lives here, making both sides easy to
    test, extend and reason about independently.

    Retry policy
    ------------
    Pass any :class:`~library.core.abstractions.IRetryPolicy` implementation
    to control retry behaviour.  The default is ``NoRetryPolicy`` (fail fast).
    Built-in policies live in ``library.retry_policies``:

    * ``NoRetryPolicy``                 – no retries (default)
    * ``FixedRetryPolicy(n)``           – retry up to *n* times immediately
    * ``ExponentialBackoffRetryPolicy`` – retry with exponential back-off

    Custom strategies only need to implement ``IRetryPolicy``; no changes to
    this class are required (Open/Closed Principle).

    Event system
    ------------
    Handlers are registered per-event with ``subscribe()``.  Any number of
    handlers may be attached to the same event; they are called in
    registration order.  Handlers receive a ``PipelineEventPayload`` so they
    can inspect context, results and errors without coupling to internals.

    Example
    -------
    >>> from library.retry_policies import FixedRetryPolicy
    >>> orchestrator = PipelineOrchestrator(context, retry_policy=FixedRetryPolicy(3))
    >>> orchestrator.subscribe(PipelineEvent.AFTER_RUN, lambda p: print(p.results))
    >>> results = orchestrator.run()
    """

    def __init__(
        self,
        context: PipelineContext,
        retry_policy: IRetryPolicy | None = None,
    ) -> None:
        self._context = context
        self._pipeline = Pipeline(context)
        self._retry_policy = retry_policy or NoRetryPolicy()
        self._handlers: dict[PipelineEvent, list[EventHandler]] = {
            e: [] for e in PipelineEvent
        }

    # ── Event API ────────────────────────────────────────────────────────────

    def subscribe(self, event: PipelineEvent, handler: EventHandler) -> None:
        """Register *handler* to be called when *event* is emitted."""
        self._handlers[event].append(handler)

    def unsubscribe(self, event: PipelineEvent, handler: EventHandler) -> None:
        """Remove a previously registered handler (no-op if not found)."""
        try:
            self._handlers[event].remove(handler)
        except ValueError:
            pass

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self) -> list[IData]:
        """
        Execute the pipeline with the configured retry and event policy.

        Returns
        -------
        list[IData]
            The results produced by each analyzer in the pipeline.

        Raises
        ------
        PipelineExecutionError
            Re-raised after the retry policy signals no further attempts.
        """
        self._emit(PipelineEvent.BEFORE_RUN, results=[])

        attempt = 1
        while True:
            try:
                results = self._pipeline.run()
                self._emit(PipelineEvent.AFTER_RUN, results=results)
                return results

            except PipelineExecutionError as exc:
                self._emit(PipelineEvent.ON_ERROR, error=exc, attempt=attempt)
                log.error("Orchestrator caught error on attempt %d: %s", attempt, exc)

                if not self._retry_policy.should_retry(attempt, exc):
                    raise

                delay = self._retry_policy.wait_seconds(attempt)
                if delay > 0:
                    log.info("Waiting %.1f s before retry…", delay)
                    time.sleep(delay)

                attempt += 1
                log.warning("Retrying pipeline (attempt %d)…", attempt)
                self._emit(PipelineEvent.ON_RETRY, attempt=attempt)

    # ── Secondary pipeline ───────────────────────────────────────────────────

    def run_secondary(self, context: PipelineContext) -> list[IData]:
        """
        Execute a secondary pipeline with a different context.

        Use this for conditional branching: run a specialised pipeline
        whose context was built by the caller based on the primary results.
        The same retry policy and event handlers are propagated.

        Example
        -------
        >>> primary_results = orchestrator.run()
        >>> if needs_keypoint_analysis(primary_results):
        ...     secondary_ctx = build_keypoint_context(primary_results)
        ...     detail = orchestrator.run_secondary(secondary_ctx)
        """
        log.info("Orchestrator: launching secondary pipeline.")
        secondary = PipelineOrchestrator(context, retry_policy=self._retry_policy)
        for event, handlers in self._handlers.items():
            for handler in handlers:
                secondary.subscribe(event, handler)
        return secondary.run()

    # ── Internals ────────────────────────────────────────────────────────────

    def _emit(
        self,
        event: PipelineEvent,
        results: list[IData] | None = None,
        error: Exception | None = None,
        attempt: int = 1,
    ) -> None:
        payload = PipelineEventPayload(
            event=event,
            pipeline=self._pipeline,
            context=self._context,
            results=results or [],
            error=error,
            attempt=attempt,
        )
        for handler in self._handlers[event]:
            try:
                handler(payload)
            except Exception as exc:
                # Handlers must NEVER crash the orchestrator
                log.warning("Event handler raised an exception (ignored): %s", exc)
