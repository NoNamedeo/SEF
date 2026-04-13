from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from library.core.abstractions.IBranchingRule import IBranchingRule
from library.core.abstractions.IData import IData
from library.core.events.DomainEvent import DomainEvent
from library.core.events.EventBus import EventBus
from library.core.pipeline.PipelineContext import PipelineContext
from library.retry_policies.NoRetryPolicy import NoRetryPolicy

if TYPE_CHECKING:
    from library.core.events.PipelineLifecycleBus import PipelineLifecycleBus

log = logging.getLogger(__name__)


class BranchingCoordinator:
    """
    Listens to domain events and spawns secondary pipelines in parallel.

    Design rationale
    ----------------
    This class extracts from PipelineOrchestrator three tightly-coupled
    responsibilities that change together but independently from retry logic:

      * **Rule evaluation** — iterating ``IBranchingRule.matches()``
      * **Parallel execution** — ``ThreadPoolExecutor`` management
      * **Future tracking** — collection, cleanup, pending count

    The coordinator is created by the builder and injected into the
    orchestrator (Dependency Inversion).  The orchestrator never touches
    rules, threads, or futures directly — it delegates.

    Lifecycle bus sharing
    ---------------------
    An optional ``PipelineLifecycleBus`` can be passed at construction.
    When present, every secondary orchestrator spawned by this coordinator
    receives the **same** bus, making secondary lifecycle events (AFTER_RUN,
    ON_ERROR …) visible to the primary's subscribers.

    Secondary pipeline policy
    -------------------------
    * **No retry** — secondary pipelines always use ``NoRetryPolicy``.
      They are speculative/corrective branches; retrying them would cause
      unbounded spawning.
    * **No handler propagation** — lifecycle handlers are shared via the
      bus, not by copying handler lists.

    Example
    -------
    >>> bus = EventBus()
    >>> coordinator = BranchingCoordinator(bus, rules=[TrackingLostBranch()])
    >>> # Orchestrator wires coordinator; domain events trigger auto-spawn
    >>> secondary_results = coordinator.collect(timeout=30)
    >>> coordinator.shutdown()
    """

    def __init__(
        self,
        event_bus: EventBus,
        rules: list[IBranchingRule],
        lifecycle_bus: PipelineLifecycleBus | None = None,
        max_workers: int = 4,
    ) -> None:
        self._event_bus = event_bus
        self._rules = list(rules)
        self._lifecycle_bus = lifecycle_bus
        self._max_workers = max_workers

        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future[list[IData]]] = []

        # Wire: every domain event is evaluated against all rules
        self._event_bus.subscribe_all(self._on_domain_event)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def event_bus(self) -> EventBus:
        """The domain EventBus this coordinator listens on."""
        return self._event_bus

    @property
    def pending_count(self) -> int:
        """Number of secondary pipelines still in flight."""
        return sum(1 for f in self._futures if not f.done())

    # ── Collection ──────────────────────────────────────────────────────────

    def collect(self, timeout: float | None = None) -> list[list[IData]]:
        """
        Wait for and return results from all auto-spawned secondary pipelines.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait for each future.  ``None`` blocks
            indefinitely.

        Returns
        -------
        list[list[IData]]
            One list of IData per successfully completed secondary pipeline.
            Failed pipelines are logged and excluded from the result.
        """
        results: list[list[IData]] = []
        for future in self._futures:
            try:
                results.append(future.result(timeout=timeout))
            except Exception as exc:
                log.error(
                    "Secondary pipeline failed: %s",
                    exc,
                    exc_info=True,
                )
        self._futures.clear()
        return results

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the internal ThreadPoolExecutor and clear pending futures.

        Safe to call multiple times.

        Parameters
        ----------
        wait:
            If ``True`` (default), block until all running pipelines finish.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
            log.info("BranchingCoordinator: ThreadPoolExecutor shut down.")
        self._futures.clear()

    # ── Internals ───────────────────────────────────────────────────────────

    def _on_domain_event(self, event: DomainEvent) -> None:
        """
        Evaluate branching rules and spawn secondary pipelines.

        Called by the EventBus for every domain event.  For each rule
        that matches, a new pipeline is submitted to the ThreadPoolExecutor.
        """
        for rule in self._rules:
            try:
                if not rule.matches(event):
                    continue

                context = rule.build_context(event)
                log.info(
                    "Branching rule %s matched event '%s' — spawning secondary pipeline.",
                    type(rule).__name__,
                    event.event_type,
                )
                self._spawn(context)

            except Exception as exc:
                log.error(
                    "Branching rule %s raised an exception (skipped): %s",
                    type(rule).__name__,
                    exc,
                    exc_info=True,
                )

    def _spawn(self, context: PipelineContext) -> Future[list[IData]]:
        """
        Submit a secondary pipeline to the ThreadPoolExecutor.

        The executor is created lazily on first spawn to avoid allocating
        threads when no branching events ever fire.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="sef-secondary-pipeline",
            )

        # Capture references for the closure
        lifecycle_bus = self._lifecycle_bus

        def _run_secondary_pipeline() -> list[IData]:
            # Lazy import to avoid circular dependency at module load time
            from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator

            secondary = PipelineOrchestrator(
                context,
                retry_policy=NoRetryPolicy(),
                lifecycle_bus=lifecycle_bus,
            )
            return secondary.run()

        future = self._executor.submit(_run_secondary_pipeline)
        self._futures.append(future)
        log.info(
            "Secondary pipeline submitted (total pending: %d).",
            self.pending_count,
        )
        return future
