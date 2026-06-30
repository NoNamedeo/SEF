from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from sef.api.execution import prepare_run_config
from sef.api.pipeline import PipelineFacade
from sef.api.registry import clone_registry, default_registry
from sef.core.events import Event, EventBus, PipelineLifecycleEvent
from sef.core.interfaces.pipeline import IBranchingRule
from sef.core.pipeline import (
    BranchingCoordinator,
    PipelineContext,
    PipelineOrchestrator,
    PipelineRunMaterializer,
)
from sef.core.plugins import PluginRegistry
from sef.core.visualization import PipelineOutputs

PipelineInput = PipelineFacade | Mapping[str, Any]
LifecycleHandler = Callable[[Event], None]

_LIFECYCLE_ALIASES = {
    "before_run": PipelineLifecycleEvent.BEFORE_RUN,
    "before": PipelineLifecycleEvent.BEFORE_RUN,
    "after_run": PipelineLifecycleEvent.AFTER_RUN,
    "after": PipelineLifecycleEvent.AFTER_RUN,
    "error": PipelineLifecycleEvent.ON_ERROR,
    "on_error": PipelineLifecycleEvent.ON_ERROR,
    "retry": PipelineLifecycleEvent.ON_RETRY,
    "on_retry": PipelineLifecycleEvent.ON_RETRY,
    "cancelled": PipelineLifecycleEvent.CANCELLED,
    "canceled": PipelineLifecycleEvent.CANCELLED,
    "rejected": PipelineLifecycleEvent.REJECTED,
    "submit_failed": PipelineLifecycleEvent.SUBMIT_FAILED,
}


@dataclass(slots=True)
class OrchestratorFacade:
    """
    Small public coordinator for running one or more already-described pipelines.

    `sef.pipeline()` describes a single pipeline. `sef.orchestrator()` executes
    pipelines, observes lifecycle events, and optionally wires branching rules.
    Declarative config intentionally remains pipeline-only; orchestration is an
    application concern and stays in Python until the usage patterns are clear.
    """

    _registry: PluginRegistry
    _include_builtins: bool = True
    _lifecycle_bus: EventBus = field(default_factory=EventBus)
    _domain_bus: EventBus = field(default_factory=EventBus)
    _orchestrator: PipelineOrchestrator | None = None
    _branching_coordinator: BranchingCoordinator | None = None

    def run(
        self,
        pipeline: PipelineInput,
        *,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> PipelineOutputs:
        """Run a pipeline synchronously through the shared orchestrator."""
        if isinstance(pipeline, PipelineContext):
            raise TypeError("orchestrator.run does not accept PipelineContext. Use run_context for advanced context execution.")
        materialized = self._materialize(
            pipeline,
            id=id,
            metadata=metadata,
            run=run,
        )
        return self._resolved_orchestrator().run_context(
            materialized.context,
            id=materialized.pipeline_id,
            metadata=materialized.execution_metadata,
            run=materialized.run_options.to_config(),
        )

    def submit(
        self,
        pipeline: PipelineInput,
        *,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> Future[PipelineOutputs]:
        """Submit a pipeline for background execution."""
        if isinstance(pipeline, PipelineContext):
            raise TypeError("orchestrator.submit does not accept PipelineContext. Use submit_context for advanced context execution.")
        materialized = self._materialize(
            pipeline,
            id=id,
            metadata=metadata,
            run=run,
        )
        return self._resolved_orchestrator().submit_context(
            materialized.context,
            id=materialized.pipeline_id,
            metadata=materialized.execution_metadata,
            run=materialized.run_options.to_config(),
        )

    def run_context(
        self,
        context: PipelineContext,
        *,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> PipelineOutputs:
        """Advanced API: run an already-materialized PipelineContext."""
        return self._resolved_orchestrator().run_context(
            context,
            id=id,
            metadata=metadata,
            run=run,
        )

    def submit_context(
        self,
        context: PipelineContext,
        *,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> Future[PipelineOutputs]:
        """Advanced API: submit an already-materialized PipelineContext."""
        return self._resolved_orchestrator().submit_context(
            context,
            id=id,
            metadata=metadata,
            run=run,
        )

    def on_lifecycle(self, event: str | PipelineLifecycleEvent, handler: LifecycleHandler) -> OrchestratorFacade:
        """
        Subscribe to a runner lifecycle event and return this facade.

        Accepted aliases include ``"before_run"``, ``"after_run"``,
        ``"error"``, ``"retry"``, ``"cancelled"``, ``"rejected"``, and
        ``"submit_failed"``.
        """
        self._lifecycle_bus.subscribe(str(_lifecycle_event(event)), handler)
        return self

    def with_branching(self, *rules: IBranchingRule | Iterable[IBranchingRule]) -> OrchestratorFacade:
        """
        Attach one or more branching rules for future runs and return this facade.

        Branching rules listen to domain events emitted by components that
        implement ``IEventEmitter``. Matching rules build child run configs,
        which are submitted through the same orchestrator.
        """
        resolved_rules = _flatten_rules(rules)
        if not resolved_rules:
            return self
        if self._branching_coordinator is None:
            self._branching_coordinator = BranchingCoordinator(
                event_bus=self._domain_bus,
                rules=resolved_rules,
                trigger_bus=self._lifecycle_bus,
            )
        else:
            self._branching_coordinator.add_rules(resolved_rules)
        return self

    def active_ids(self) -> list[str]:
        """Return active or queued pipeline ids."""
        return self._resolved_orchestrator().active_ids()

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the underlying runner."""
        self._resolved_orchestrator().shutdown(wait=wait)

    def _resolved_orchestrator(self) -> PipelineOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = PipelineOrchestrator(
                registry=self._registry,
                bus=self._lifecycle_bus,
                domain_bus=self._domain_bus,
            )
        return self._orchestrator

    def _materialize(
        self,
        pipeline: PipelineInput,
        *,
        id: str | None,
        metadata: Mapping[str, Any] | None,
        run: Mapping[str, Any] | None,
    ):
        prepared = prepare_run_config(
            pipeline,
            id=id,
            metadata=metadata,
            run=run,
            registry=self._registry,
            include_builtins=self._include_builtins,
        )
        return PipelineRunMaterializer(prepared.registry).materialize(prepared.config)


def orchestrator(
    *,
    registry: PluginRegistry | None = None,
    include_builtins: bool = True,
) -> OrchestratorFacade:
    """
    Create the high-level orchestration facade.

    Use this when a run needs lifecycle observation, async submission, or
    event-driven branching. Use ``sef.core`` directly for custom runners,
    monitors, stores, retry policies, or framework integrations.
    """
    return OrchestratorFacade(
        _registry=clone_registry(registry) if registry is not None else default_registry(include_builtins=include_builtins),
        _include_builtins=include_builtins,
    )


def _lifecycle_event(event: str | PipelineLifecycleEvent) -> PipelineLifecycleEvent:
    if isinstance(event, PipelineLifecycleEvent):
        return event
    value = str(event).strip()
    for lifecycle_event in PipelineLifecycleEvent:
        if value == str(lifecycle_event):
            return lifecycle_event
    normalized = value.lower().replace("-", "_")
    if normalized.startswith("pipeline."):
        normalized = normalized.removeprefix("pipeline.")
    try:
        return _LIFECYCLE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(_LIFECYCLE_ALIASES))
        raise ValueError(f"Unknown pipeline lifecycle event '{event}'. Expected one of: {allowed}.") from exc


def _flatten_rules(rules: tuple[IBranchingRule | Iterable[IBranchingRule], ...]) -> list[IBranchingRule]:
    flattened: list[IBranchingRule] = []
    for item in rules:
        if isinstance(item, IBranchingRule):
            flattened.append(item)
            continue
        if isinstance(item, Iterable):
            for rule in item:
                _append_rule(flattened, rule)
            continue
        _append_rule(flattened, item)
    return flattened


def _append_rule(rules: list[IBranchingRule], rule: object) -> None:
    if not isinstance(rule, IBranchingRule):
        raise TypeError("with_branching expects IBranchingRule instances.")
    rules.append(rule)
