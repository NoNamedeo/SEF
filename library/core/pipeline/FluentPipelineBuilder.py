from __future__ import annotations

from collections.abc import Iterable

from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameCaptureConfig
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.VisualizerBinding import VisualizerBinding


class FluentPipelineBuilder:
    """
    Programmatic builder for PipelineContext.

    The builder only collects pipeline components and creates a validated
    context. Execution belongs to PipelineOrchestrator.
    """

    def __init__(self) -> None:
        self._frame_extractor: IFrameExtractor | None = None
        self._signal_extractor: ISignalExtractor | None = None
        self._frame_processors: list[IFrameBufferProcessor] = []
        self._signal_cleaners: list[ISignalCleaner] = []
        self._analyzers: list[IAnalyzer] = []
        self._visualizers: list[IVisualizer] = []
        self._visualizer_bindings: list[VisualizerBinding] = []
        self._intermediate_frame_capture = IntermediateFrameCaptureConfig.disabled()
        self._intermediate_frame_visualizers: list[IVisualizer] = []

    # ── Pipeline components ─────────────────────────────────────────────────

    def with_frame_extractor(self, extractor: IFrameExtractor) -> FluentPipelineBuilder:
        self._frame_extractor = extractor
        return self

    def with_signal_extractor(self, extractor: ISignalExtractor) -> FluentPipelineBuilder:
        self._signal_extractor = extractor
        return self

    def with_frame_processors(self, processors: Iterable[IFrameBufferProcessor]) -> FluentPipelineBuilder:
        self._frame_processors = list(processors)
        return self

    def add_frame_processor(self, processor: IFrameBufferProcessor) -> FluentPipelineBuilder:
        self._frame_processors.append(processor)
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

    def with_visualizer_bindings(
        self,
        bindings: Iterable[VisualizerBinding],
    ) -> FluentPipelineBuilder:
        self._visualizer_bindings = list(bindings)
        return self

    def add_visualizer_for_results(
        self,
        visualizer: IVisualizer,
        result_indices: Iterable[int],
    ) -> FluentPipelineBuilder:
        self._visualizer_bindings.append(
            VisualizerBinding(visualizer=visualizer, result_indices=tuple(result_indices))
        )
        return self

    def with_intermediate_frame_capture(
        self,
        config: IntermediateFrameCaptureConfig | dict | None,
    ) -> FluentPipelineBuilder:
        """Enable or disable bounded intermediate frame capture."""
        self._intermediate_frame_capture = (
            config
            if isinstance(config, IntermediateFrameCaptureConfig)
            else IntermediateFrameCaptureConfig.from_mapping(config)
        )
        return self

    def with_intermediate_frame_visualizers(
        self,
        visualizers: Iterable[IVisualizer],
    ) -> FluentPipelineBuilder:
        """Replace visualizers dedicated to intermediate frame collections."""
        self._intermediate_frame_visualizers = list(visualizers)
        return self

    def add_intermediate_frame_visualizer(self, visualizer: IVisualizer) -> FluentPipelineBuilder:
        """Add a visualizer dedicated to intermediate frame collections."""
        self._intermediate_frame_visualizers.append(visualizer)
        return self

    # ── Context helper ──────────────────────────────────────────────────────

    def build_context(self) -> PipelineContext:
        """Build and return the PipelineContext from the configured components."""
        return PipelineContext(
            frame_extractor=self._frame_extractor,
            signal_extractor=self._signal_extractor,
            frame_processors=list(self._frame_processors),
            signal_cleaners=list(self._signal_cleaners),
            analyzers=list(self._analyzers),
            visualizers=list(self._visualizers),
            visualizer_bindings=list(self._visualizer_bindings),
            intermediate_frame_capture=self._intermediate_frame_capture,
            intermediate_frame_visualizers=list(self._intermediate_frame_visualizers),
        )
