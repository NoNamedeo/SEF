from __future__ import annotations

import logging
import threading

from library.core.events.Event import Event
from library.core.events.PipelineEvent import PipelineEvent
from library.core.interfaces.pipeline.IBranchingRule import IBranchingRule
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

    When a rule matches an Event the coordinator builds the secondary
    PipelineContext via IBranchingRule.build_context() and dispatches a
    PipelineEvent onto the trigger bus. The PipelineOrchestrator subscribed
    to that bus picks it up and delegates execution through its monitor and
    runner.

    Rule isolation
    --------------
    A rule that raises during ``matches()`` or ``build_context()`` is logged
    and skipped; remaining rules still run.
    """

    def __init__(
        self,
        event_bus: IEventBus,
        rules: list[IBranchingRule],
        trigger_bus: IEventBus,
    ) -> None:
        self._event_bus = event_bus
        self._rules = list(rules)
        self._trigger_bus = trigger_bus
        self._counter = 0
        self._lock = threading.Lock()
        self._event_bus.subscribe(IEventBus.WILDCARD, self._on_domain_event)

    @property
    def event_bus(self) -> IEventBus:
        return self._event_bus

    def _on_domain_event(self, event: Event) -> None:
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
                    PipelineEvent.create(
                        pipeline_id=pipeline_id,
                        context=context,
                        source=type(self).__name__,
                        correlation_id=event.correlation_id or event.event_id,
                        execution_metadata={
                            "parent_pipeline_id": str(event.payload.get("pipeline_id", "")) or None,
                            "branch_rule": type(rule).__name__,
                            "trigger_event_type": event.event_type,
                            "trigger_source": event.source,
                        },
                    )
                )
            except Exception as exc:
                self._trigger_bus.dispatch(
                    Event(
                        event_type="pipeline.branching_failed",
                        source=type(self).__name__,
                        correlation_id=event.correlation_id or event.event_id,
                        payload={
                            "pipeline_id": str(event.payload.get("pipeline_id", "-")),
                            "rule": type(rule).__name__,
                            "trigger_event_type": event.event_type,
                            "error": str(exc),
                        },
                    )
                )
                log.error(
                    "Branching rule %s raised (skipped): %s",
                    type(rule).__name__,
                    exc,
                    exc_info=True,
                )
