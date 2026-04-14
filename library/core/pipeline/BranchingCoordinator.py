from __future__ import annotations

import logging
import threading

from library.core.interfaces.pipeline.IBranchingRule import IBranchingRule
from library.core.events.DomainEvent import DomainEvent
from library.core.events.EventBus import EventBus
from library.core.interfaces.pipeline.IEventBus import IEventBus

log = logging.getLogger(__name__)


class BranchingCoordinator:
    """
    Evaluates IBranchingRule strategies against domain events and dispatches
    a PipelineEvent trigger for each match.

    Design rationale
    ----------------
    The coordinator is the single place that couples domain events to pipeline
    spawning.  It has no thread pool, no futures, and no retry logic — those
    responsibilities belong to IPipelineRunner and IRetryPolicy respectively.

    When a rule matches a DomainEvent the coordinator builds the secondary
    PipelineContext via IBranchingRule.build_context() and dispatches a
    PipelineEvent onto the trigger bus.  The PipelineOrchestrator subscribed
    to that bus picks it up and delegates execution through its own chain
    (IPipelineBuilder → IPipelineMonitor → IPipelineRunner).

    Rule isolation
    --------------
    A rule that raises during ``matches()`` or ``build_context()`` is logged
    and skipped; remaining rules still run.
    """

    def __init__(
        self,
        event_bus: EventBus,
        rules: list[IBranchingRule],
        trigger_bus: IEventBus,
    ) -> None:
        self._event_bus = event_bus
        self._rules = list(rules)
        self._trigger_bus = trigger_bus
        self._counter = 0
        self._lock = threading.Lock()
        self._event_bus.subscribe_all(self._on_domain_event)

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    def _on_domain_event(self, event: DomainEvent) -> None:
        from library.core.artifacts.PipelineEvent import PipelineEvent

        for rule in self._rules:
            try:
                if not rule.matches(event):
                    continue
                context = rule.build_context(event)
                with self._lock:
                    self._counter += 1
                    pipeline_id = f"secondary-{self._counter}"
                log.info(
                    "Rule %s matched '%s' — dispatching %s.",
                    type(rule).__name__,
                    event.event_type,
                    pipeline_id,
                )
                self._trigger_bus.dispatch(
                    PipelineEvent(pipeline_id=pipeline_id, context=context)
                )  # noqa: E501
            except Exception as exc:
                log.error(
                    "Branching rule %s raised (skipped): %s",
                    type(rule).__name__,
                    exc,
                    exc_info=True,
                )
