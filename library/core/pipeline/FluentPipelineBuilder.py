from __future__ import annotations

from collections.abc import Iterable

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.abstractions.IFrameExtractor import IFrameExtractor
from library.core.abstractions.ISignalCleaner import ISignalCleaner
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.abstractions.IVisualizer import IVisualizer
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator


class FluentPipelineBuilder:
    """
    Programmatic, type-safe builder for PipelineOrchestrator.

    Design rationale
    ----------------
    Pipeline is an internal execution detail — callers never construct or
    run it directly.  The only public product of this builder is a
    PipelineOrchestrator, which provides retry logic, lifecycle events and
    secondary-pipeline support out of the box.

    Validation
    ----------
    Deferred to build(): the builder can be populated incrementally
    (e.g. in test fixtures) without raising on every setter call.
    Missing required components produce a ValueError with a clear message
    listing exactly what is absent.

    Usage
    -----
    >>> orchestrator = (
    ...     FluentPipelineBuilder()
    ...     .with_frame_extractor(OpenCVBufferedFrameExtractor(path))
    ...     .with_signal_extractor(OpenCVBufferedSignalExtractor(roi))
    ...     .add_analyzer(VerticalPositionAnalyzer())
    ...     .build()
    ... )
    >>> results = orchestrator.run()
    """

    def __init__(self) -> None:
        self._frame_extractor:  IFrameExtractor  | None = None
        self._signal_extractor: ISignalExtractor | None = None
        self._frame_cleaners:   list[IFrameCleaner]     = []
        self._signal_cleaners:  list[ISignalCleaner]    = []
        self._analyzers:        list[IAnalyzer]         = []
        self._visualizers:      list[IVisualizer]       = []

    # ── Setters (return self for fluent chaining) ────────────────────────────

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

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, max_retries: int = 0) -> PipelineOrchestrator:
        """
        Validate and return a fully configured PipelineOrchestrator.

        Parameters
        ----------
        max_retries : int
            Number of additional execution attempts on PipelineExecutionError.
            0 (default) means no retries — fail immediately on first error.

        Raises
        ------
        ValueError
            If frame_extractor, signal_extractor or at least one analyzer
            have not been provided.
        """
        self._validate()
        context = PipelineContext(
            frame_extractor  = self._frame_extractor,
            signal_extractor = self._signal_extractor,
            frame_cleaners   = list(self._frame_cleaners),
            signal_cleaners  = list(self._signal_cleaners),
            analyzers        = list(self._analyzers),
            visualizers      = list(self._visualizers),
        )
        return PipelineOrchestrator(context, max_retries=max_retries)

    # ── Internals ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        missing: list[str] = []
        if self._frame_extractor is None:
            missing.append("frame_extractor")
        if self._signal_extractor is None:
            missing.append("signal_extractor")
        if not self._analyzers:
            missing.append("analyzers (at least one required)")
        if missing:
            raise ValueError(
                f"FluentPipelineBuilder.build(): missing required components: "
                f"{', '.join(missing)}"
            )