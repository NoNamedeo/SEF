from __future__ import annotations

import logging
import time

from library.core.abstractions.IData import IData
from library.core.abstractions.IRetryPolicy import IRetryPolicy
from library.core.events.EventBus import EventBus
from library.core.events.PipelineLifecycleBus import (
    EventHandler,
    LifecycleEventHandler,
    PipelineEvent,
    PipelineEventPayload,
    PipelineLifecycleBus,
)
from library.core.pipeline.BranchingCoordinator import BranchingCoordinator
from library.core.pipeline.Pipeline import Pipeline, PipelineExecutionError
from library.core.pipeline.PipelineContext import PipelineContext
from library.retry_policies.NoRetryPolicy import NoRetryPolicy

log = logging.getLogger(__name__)


# Re-export for backward compatibility — these types used to live here.
__all__ = [
    "PipelineOrchestrator",
    "PipelineEvent",
    "PipelineEventPayload",
    "PipelineLifecycleBus",
    "LifecycleEventHandler",
    "EventHandler",
]


class PipelineOrchestrator:
    """
    Facade that coordinates pipeline execution.

    Design rationale
    ----------------
    The orchestrator is the single entry-point for running a pipeline.
    Internally it delegates each concern to a specialised component:

    +--------------------------+----------------------------------------------+
    | Concern                  | Component                                    |
    +==========================+==============================================+
    | Retry policy             | ``IRetryPolicy`` (Strategy)                  |
    | Lifecycle events         | ``PipelineLifecycleBus`` (Observer)          |
    | Branching + parallelism  | ``BranchingCoordinator`` (mediator)          |
    | Step execution           | ``Pipeline`` (executor)                      |
    +--------------------------+----------------------------------------------+

    This decomposition respects the **Single Responsibility Principle**:
    each class has exactly one reason to change.

    The orchestrator itself is a **Facade**: it exposes convenience methods
    (``subscribe``, ``collect_secondary_results``, ``shutdown``) that
    delegate to the owned components, keeping the public API compact.

    Lifecycle bus sharing
    ---------------------
    The ``PipelineLifecycleBus`` is injectable.  When the same bus is
    passed to both the primary orchestrator and the ``BranchingCoordinator``,
    lifecycle events from secondary pipelines (BEFORE_RUN, AFTER_RUN,
    ON_ERROR) arrive at the same subscribers as the primary's events.

    This solves the original limitation where secondary pipelines were
    invisible to the lifecycle event system.

    Example
    -------
    >>> from library.retry_policies import FixedRetryPolicy
    >>> orchestrator = PipelineOrchestrator(context, retry_policy=FixedRetryPolicy(3))
    >>> orchestrator.subscribe(PipelineEvent.AFTER_RUN, lambda p: print(p.results))
    >>> results = orchestrator.run()

    Example with branching (via builder)
    -------------------------------------
    >>> orchestrator = (
    ...     FluentPipelineBuilder()
    ...     .with_frame_extractor(...)
    ...     .with_signal_extractor(EventAwareTracker(...))
    ...     .add_analyzer(...)
    ...     .add_branching_rule(TrackingLostBranch())
    ...     .build()
    ... )
    >>> primary = orchestrator.run()
    >>> secondary = orchestrator.collect_secondary_results(timeout=30)
    >>> orchestrator.shutdown()
    """

    def __init__(
        self,
        context: PipelineContext,
        retry_policy: IRetryPolicy | None = None,
        lifecycle_bus: PipelineLifecycleBus | None = None,
        branching: BranchingCoordinator | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._context = context
        self._retry_policy = retry_policy or NoRetryPolicy()
        self._lifecycle_bus = lifecycle_bus or PipelineLifecycleBus()
        self._branching = branching

        # Resolve EventBus: explicit > from coordinator > None
        effective_bus = event_bus
        if effective_bus is None and branching is not None:
            effective_bus = branching.event_bus

        self._pipeline = Pipeline(context, event_bus=effective_bus)

    # ── Lifecycle bus delegation ────────────────────────────────────────────

    @property
    def lifecycle_bus(self) -> PipelineLifecycleBus:
        """The lifecycle event bus used by this orchestrator."""
        return self._lifecycle_bus

    def subscribe(self, event: PipelineEvent, handler: LifecycleEventHandler) -> None:
        """Register *handler* to be called when *event* is emitted."""
        self._lifecycle_bus.subscribe(event, handler)

    def unsubscribe(self, event: PipelineEvent, handler: LifecycleEventHandler) -> None:
        """Remove a previously registered handler (no-op if not found)."""
        self._lifecycle_bus.unsubscribe(event, handler)

    # ── Domain EventBus access ──────────────────────────────────────────────

    @property
    def event_bus(self) -> EventBus | None:
        """Return the domain EventBus, or None if branching is not configured."""
        if self._branching is not None:
            return self._branching.event_bus
        return None

    # ── Run ─────────────────────────────────────────────────────────────────

    def run(self) -> list[IData]:
        """
        Execute the pipeline with the configured retry and event policy.

        During execution, if any component emits domain events and
        a ``BranchingCoordinator`` is configured, secondary pipelines
        are automatically spawned in parallel.

        After ``run()`` returns, call ``collect_secondary_results()`` to
        retrieve results from any auto-spawned secondary pipelines.

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
                    log.info("Waiting %.1f s before retry...", delay)
                    time.sleep(delay)

                attempt += 1
                log.warning("Retrying pipeline (attempt %d)...", attempt)
                self._emit(PipelineEvent.ON_RETRY, attempt=attempt)

    # ── Secondary pipelines (manual) ────────────────────────────────────────

    def run_secondary(self, context: PipelineContext) -> list[IData]:
        """
        Execute a secondary pipeline with a different context (synchronous).

        The same ``lifecycle_bus`` is shared so that lifecycle events from
        the secondary pipeline are visible to handlers registered on the
        primary orchestrator.
        """
        log.info("Orchestrator: launching secondary pipeline (manual).")
        secondary = PipelineOrchestrator(
            context,
            retry_policy=NoRetryPolicy(),
            lifecycle_bus=self._lifecycle_bus,
        )
        return secondary.run()

    # ── Secondary pipelines (auto / parallel) delegation ────────────────────

    def collect_secondary_results(
        self,
        timeout: float | None = None,
    ) -> list[list[IData]]:
        """
        Wait for and return results from all auto-spawned secondary pipelines.

        Delegates to ``BranchingCoordinator.collect()``.
        Returns an empty list if no coordinator is configured.
        """
        if self._branching is None:
            return []
        return self._branching.collect(timeout)

    @property
    def pending_secondary_count(self) -> int:
        """Number of secondary pipelines still in flight."""
        if self._branching is None:
            return 0
        return self._branching.pending_count

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the BranchingCoordinator's ThreadPoolExecutor.

        Safe to call multiple times.  No-op if no coordinator is configured.
        """
        if self._branching is not None:
            self._branching.shutdown(wait)

    # ── Internals ───────────────────────────────────────────────────────────

    def _emit(
        self,
        event: PipelineEvent,
        results: list[IData] | None = None,
        error: Exception | None = None,
        attempt: int = 1,
    ) -> None:
        payload = PipelineEventPayload(
            event=event,
            context=self._context,
            results=results or [],
            error=error,
            attempt=attempt,
            pipeline=self._pipeline,
        )
        self._lifecycle_bus.emit(payload)
