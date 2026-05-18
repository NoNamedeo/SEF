from __future__ import annotations

from typing import Any

from library.core.interfaces.StreamingContracts import (
    IStreamingAnalyzer,
    IStreamingFrameBufferProcessor,
    IStreamingFrameExporter,
    IStreamingFrameExtractor,
    IStreamingSignalCleaner,
    IStreamingSignalExtractor,
    IStreamingVisualizer,
)
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineExecutionPlan import capabilities_of


class PipelineComponentCapabilities:
    """
    Centralizes runtime capability checks for adaptive execution.

    The same rules decide whether a stage may stream, where the pipeline must
    materialize, and whether the signal tail can run concurrently. Centralizing
    the checks prevents the execution path and planning path from drifting.

    Rule
    ----
    A component is streamable only when both conditions are true:
    - it implements the matching streaming interface;
    - its ``StageCapabilities`` explicitly says it supports streaming and does
      not require a complete sequence.

    Visualizers are the exception: they currently expose streaming support by
    implementing ``IStreamingVisualizer``.
    """

    @staticmethod
    def can_stream_frame_extractor(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingFrameExtractor) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def can_stream_frame_processor(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingFrameBufferProcessor) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def can_stream_frame_exporter(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingFrameExporter) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def can_stream_signal_extractor(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingSignalExtractor) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def can_stream_signal_cleaner(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingSignalCleaner) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def can_stream_analyzer(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingAnalyzer) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def can_stream_visualizer(component: Any) -> bool:
        return isinstance(component, IStreamingVisualizer)

    @classmethod
    def can_stream_frame_exporters(cls, context: PipelineContext) -> bool:
        return all(cls.can_stream_frame_exporter(exporter) for exporter in context.frame_exporters)

    @classmethod
    def can_stream_signal_tail(cls, context: PipelineContext) -> bool:
        visualizers = [*context.visualizers, *(binding.visualizer for binding in context.visualizer_bindings)]
        return (
            cls.can_stream_signal_extractor(context.signal_extractor)
            and all(cls.can_stream_signal_cleaner(cleaner) for cleaner in context.signal_cleaners)
            and all(cls.can_stream_analyzer(analyzer) for analyzer in context.analyzers)
            and all(isinstance(visualizer, IStreamingVisualizer) for visualizer in visualizers)
        )
