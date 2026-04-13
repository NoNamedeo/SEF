from __future__ import annotations

from collections.abc import Iterable

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.IBranchingRule import IBranchingRule
from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.abstractions.IFrameExtractor import IFrameExtractor
from library.core.abstractions.IRetryPolicy import IRetryPolicy
from library.core.abstractions.ISignalCleaner import ISignalCleaner
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.abstractions.IVisualizer import IVisualizer
from library.core.events.EventBus import EventBus
from library.core.events.PipelineLifecycleBus import PipelineLifecycleBus
from library.core.pipeline.BranchingCoordinator import BranchingCoordinator
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.validators.pipeline import PipelineContextValidator
from library.retry_policies.FixedRetryPolicy import FixedRetryPolicy
from library.retry_policies.NoRetryPolicy import NoRetryPolicy


class FluentPipelineBuilder:
    """
    Programmatic, type-safe builder for PipelineOrchestrator.

    Design rationale
    ----------------
    Pipeline is an internal execution detail — callers never construct or
    run it directly.  The only public product of this builder is a
    ``PipelineOrchestrator``, which provides retry logic, lifecycle events,
    domain-event-driven branching, and parallel secondary-pipeline support.

    ``build()`` takes **no parameters**: all configuration is done via
    fluent ``.with_*()`` / ``.add_*()`` methods.  This is the pure Builder
    pattern — ``build()`` only creates the product, it never configures it.

    Validation
    ----------
    Deferred to ``build()``: the builder can be populated incrementally
    (e.g. in test fixtures) without raising on every setter call.
    Missing required components produce a ``ValueError`` with a clear
    message listing exactly what is absent.

    Usage
    -----
    >>> orchestrator = (
    ...     FluentPipelineBuilder()
    ...     .with_frame_extractor(OpenCVBufferedFrameExtractor(path))
    ...     .with_signal_extractor(OpenCVBufferedSignalExtractor(roi))
    ...     .add_analyzer(VerticalPositionAnalyzer())
    ...     .with_max_retries(2)
    ...     .build()
    ... )
    >>> results = orchestrator.run()

    Usage with branching
    --------------------
    >>> orchestrator = (
    ...     FluentPipelineBuilder()
    ...     .with_frame_extractor(...)
    ...     .with_signal_extractor(...)
    ...     .add_analyzer(...)
    ...     .add_branching_rule(TrackingLostBranch())
    ...     .build()
    ... )
    >>> primary = orchestrator.run()
    >>> secondary = orchestrator.collect_secondary_results()
    >>> orchestrator.shutdown()
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
        self._lifecycle_bus: PipelineLifecycleBus | None = None
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
        """
        Set a custom retry policy (full control).

        Mutually exclusive with ``with_max_retries()`` — the last call wins.
        """
        self._retry_policy = policy
        return self

    def with_max_retries(self, n: int) -> FluentPipelineBuilder:
        """
        Convenience: set retry policy from a retry count.

        * ``n > 0`` → ``FixedRetryPolicy(n)``
        * ``n == 0`` → ``NoRetryPolicy()``

        Mutually exclusive with ``with_retry_policy()`` — the last call wins.
        """
        self._retry_policy = FixedRetryPolicy(n) if n > 0 else NoRetryPolicy()
        return self

    # ── Branching & events ──────────────────────────────────────────────────

    def with_event_bus(self, bus: EventBus) -> FluentPipelineBuilder:
        """
        Inject a custom domain EventBus.

        If not set, the builder creates one automatically when branching
        rules are configured.
        """
        self._event_bus = bus
        return self

    def add_branching_rule(self, rule: IBranchingRule) -> FluentPipelineBuilder:
        """Add a branching rule for automatic event-driven pipeline spawning."""
        self._branching_rules.append(rule)
        return self

    def with_branching_rules(self, rules: Iterable[IBranchingRule]) -> FluentPipelineBuilder:
        self._branching_rules = list(rules)
        return self

    # ── Lifecycle & workers ─────────────────────────────────────────────────

    def with_lifecycle_bus(self, bus: PipelineLifecycleBus) -> FluentPipelineBuilder:
        """
        Inject a shared lifecycle bus.

        When set, the same bus is passed to both the primary orchestrator
        and the ``BranchingCoordinator``, making secondary-pipeline
        lifecycle events visible to the primary's subscribers.
        """
        self._lifecycle_bus = bus
        return self

    def with_max_workers(self, n: int) -> FluentPipelineBuilder:
        """Maximum number of threads in the BranchingCoordinator's pool."""
        self._max_workers = n
        return self

    # ── Build ───────────────────────────────────────────────────────────────

    def build(self) -> PipelineOrchestrator:
        """
        Validate and return a fully configured ``PipelineOrchestrator``.

        All configuration must be done via fluent methods *before* calling
        ``build()``.  This method takes **no parameters** — it only
        assembles the product.

        Raises
        ------
        ValueError
            If the validation is failed.
        """
        context = PipelineContext(
            frame_extractor=self._frame_extractor,
            signal_extractor=self._signal_extractor,
            frame_cleaners=list(self._frame_cleaners),
            signal_cleaners=list(self._signal_cleaners),
            analyzers=list(self._analyzers),
            visualizers=list(self._visualizers),
        )
        PipelineContextValidator.validate(context)

        retry_policy = self._retry_policy or NoRetryPolicy()
        lifecycle_bus = self._lifecycle_bus

        # ── Assemble BranchingCoordinator if rules are present ──────────
        event_bus = self._event_bus
        branching: BranchingCoordinator | None = None

        if self._branching_rules:
            if event_bus is None:
                event_bus = EventBus()
            branching = BranchingCoordinator(
                event_bus=event_bus,
                rules=list(self._branching_rules),
                lifecycle_bus=lifecycle_bus,
                max_workers=self._max_workers,
            )

        return PipelineOrchestrator(
            context,
            retry_policy=retry_policy,
            lifecycle_bus=lifecycle_bus,
            branching=branching,
            event_bus=event_bus,
        )
