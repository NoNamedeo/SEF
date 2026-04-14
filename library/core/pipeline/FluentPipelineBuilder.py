from __future__ import annotations

from collections.abc import Iterable

from library.core.events.EventBus import EventBus
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.interfaces.pipeline.IBranchingRule import IBranchingRule
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.interfaces.pipeline.IRetryPolicy import IRetryPolicy
from library.core.pipeline.BranchingCoordinator import BranchingCoordinator
from library.core.pipeline.DefaultPipelineBuilder import DefaultPipelineBuilder
from library.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner
from library.retry_policies.NoRetryPolicy import NoRetryPolicy


class FluentPipelineBuilder:
    """
    Programmatic builder for PipelineOrchestrator and PipelineContext.

    ``build()`` takes **no parameters**: all configuration is done via fluent
    ``.with_*()`` / ``.add_*()`` methods.

    Trigger bus
    -----------
    The caller must supply a trigger bus via ``.with_trigger_bus()`` — or
    accept the auto-created one — and keep a reference to it.  Pipeline
    execution is initiated by dispatching a ``PipelineEvent`` onto that bus:

        bus = EventBus()
        orchestrator = FluentPipelineBuilder().with_trigger_bus(bus)...build()
        bus.dispatch(PipelineEvent("run-1", context))

    Event buses
    -----------
    * ``with_trigger_bus`` — receives ``PipelineEvent`` triggers and
      ``PipelineLifecyclePayload`` lifecycle events.
    * ``with_event_bus``   — domain EventBus injected into IEventEmitter
      components; also consumed by BranchingCoordinator.

    Both default to auto-created EventBus instances if not provided.
    """

    def __init__(self) -> None:
        self._frame_extractor: IFrameExtractor | None = None
        self._signal_extractor: ISignalExtractor | None = None
        self._frame_cleaners: list[IFrameCleaner] = []
        self._signal_cleaners: list[ISignalCleaner] = []
        self._analyzers: list[IAnalyzer] = []
        self._visualizers: list[IVisualizer] = []
        self._branching_rules: list[IBranchingRule] = []
        self._retry_policy: IRetryPolicy | None = None
        self._event_bus: EventBus | None = None
        self._trigger_bus: IEventBus | None = None
        self._max_workers: int = 4

    # ── Pipeline components ─────────────────────────────────────────────────

    def with_frame_extractor(self, extractor: IFrameExtractor) -> FluentPipelineBuilder:
        self._frame_extractor = extractor
        return self

    def with_signal_extractor(self, extractor: ISignalExtractor) -> FluentPipelineBuilder:
        self._signal_extractor = extractor
        return self

    def with_frame_cleaners(self, cleaners: Iterable[IFrameCleaner]) -> FluentPipelineBuilder:
        self._frame_cleaners = list(cleaners)
        return self

    def add_frame_cleaner(self, cleaner: IFrameCleaner) -> FluentPipelineBuilder:
        self._frame_cleaners.append(cleaner)
        return self

    def with_signal_cleaners(self, cleaners: Iterable[ISignalCleaner]) -> FluentPipelineBuilder:
        self._signal_cleaners = list(cleaners)
        return self

    def add_signal_cleaner(self, cleaner: ISignalCleaner) -> FluentPipelineBuilder:
        self._signal_cleaners.append(cleaner)
        return self

    def with_analyzers(self, analyzers: Iterable[IAnalyzer]) -> FluentPipelineBuilder:
        self._analyzers = list(analyzers)
        return self

    def add_analyzer(self, analyzer: IAnalyzer) -> FluentPipelineBuilder:
        self._analyzers.append(analyzer)
        return self

    def with_visualizers(self, visualizers: Iterable[IVisualizer]) -> FluentPipelineBuilder:
        self._visualizers = list(visualizers)
        return self

    def add_visualizer(self, visualizer: IVisualizer) -> FluentPipelineBuilder:
        self._visualizers.append(visualizer)
        return self

    # ── Retry policy ────────────────────────────────────────────────────────

    def with_retry_policy(self, policy: IRetryPolicy) -> FluentPipelineBuilder:
        self._retry_policy = policy
        return self

    # ── Event buses ─────────────────────────────────────────────────────────

    def with_event_bus(self, bus: EventBus) -> FluentPipelineBuilder:
        """Inject a custom domain EventBus for IEventEmitter components."""
        self._event_bus = bus
        return self

    def with_trigger_bus(self, bus: IEventBus) -> FluentPipelineBuilder:
        """
        Inject the trigger bus.

        The same bus receives PipelineEvent triggers (dispatched by the
        caller to start a pipeline) and PipelineLifecyclePayload events
        (dispatched by ThreadedPipelineRunner during execution).
        """
        self._trigger_bus = bus
        return self

    # ── Branching ───────────────────────────────────────────────────────────

    def add_branching_rule(self, rule: IBranchingRule) -> FluentPipelineBuilder:
        self._branching_rules.append(rule)
        return self

    def with_branching_rules(self, rules: Iterable[IBranchingRule]) -> FluentPipelineBuilder:
        self._branching_rules = list(rules)
        return self

    def with_max_workers(self, n: int) -> FluentPipelineBuilder:
        self._max_workers = n
        return self

    # ── Context helper ──────────────────────────────────────────────────────

    def build_context(self) -> PipelineContext:
        """
        Build and return the PipelineContext from the configured components.

        Use this to create the context to embed in a PipelineEvent trigger:

            context = builder.build_context()
            bus.dispatch(PipelineEvent("run-1", context))
        """
        return PipelineContext(
            frame_extractor=self._frame_extractor,
            signal_extractor=self._signal_extractor,
            frame_cleaners=list(self._frame_cleaners),
            signal_cleaners=list(self._signal_cleaners),
            analyzers=list(self._analyzers),
            visualizers=list(self._visualizers),
        )

    # ── Build ───────────────────────────────────────────────────────────────

    def build(self) -> PipelineOrchestrator:
        """
        Wire and return a fully configured PipelineOrchestrator.

        Takes **no parameters** — all configuration via fluent methods.
        """
        trigger_bus: IEventBus = self._trigger_bus or EventBus()
        monitor = InMemoryPipelineMonitor()

        domain_bus = self._event_bus
        if self._branching_rules and domain_bus is None:
            domain_bus = EventBus()

        pipeline_builder = DefaultPipelineBuilder(domain_bus=domain_bus)

        runner = ThreadedPipelineRunner(
            monitor=monitor,
            retry_policy=self._retry_policy or NoRetryPolicy(),
            lifecycle_bus=trigger_bus,
            max_workers=self._max_workers,
        )

        self._branching_coordinator = None
        if self._branching_rules and domain_bus is not None:
            self._branching_coordinator = BranchingCoordinator(
                event_bus=domain_bus,
                rules=list(self._branching_rules),
                trigger_bus=trigger_bus,
            )

        return PipelineOrchestrator(
            builder=pipeline_builder,
            runner=runner,
            monitor=monitor,
            bus=trigger_bus,
        )
