from __future__ import annotations

from typing import Any

from sef.core.interfaces.StreamingContracts import (
    IStreamingAnalyzer,
    IStreamingFrameBufferProcessor,
    IStreamingFrameExporter,
    IStreamingFrameExtractor,
    IStreamingSignalCleaner,
    IStreamingSignalExtractor,
    IStreamingVisualizer,
)
from sef.core.pipeline.PipelineExecutionPlan import capabilities_of


class PipelineComponentCapabilities:
    """
    Centralizes runtime capability checks for adaptive execution.

    The same rules decide whether a stage may stream and where the pipeline
    must materialize. Centralizing the checks prevents the execution path and
    planning path from drifting.

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
        return PipelineComponentCapabilities._supports_streaming_contract(
            component,
            IStreamingFrameExtractor,
        )

    @staticmethod
    def can_stream_frame_processor(component: Any) -> bool:
        return PipelineComponentCapabilities._supports_streaming_contract(
            component,
            IStreamingFrameBufferProcessor,
        )

    @staticmethod
    def can_stream_frame_exporter(component: Any) -> bool:
        return PipelineComponentCapabilities._supports_streaming_contract(
            component,
            IStreamingFrameExporter,
        )

    @staticmethod
    def can_stream_signal_extractor(component: Any) -> bool:
        return PipelineComponentCapabilities._supports_streaming_contract(
            component,
            IStreamingSignalExtractor,
        )

    @staticmethod
    def can_stream_signal_cleaner(component: Any) -> bool:
        return PipelineComponentCapabilities._supports_streaming_contract(
            component,
            IStreamingSignalCleaner,
        )

    @staticmethod
    def can_stream_analyzer(component: Any) -> bool:
        return PipelineComponentCapabilities._supports_streaming_contract(
            component,
            IStreamingAnalyzer,
        )

    @staticmethod
    def can_stream_visualizer(component: Any) -> bool:
        return isinstance(component, IStreamingVisualizer)

    @staticmethod
    def _supports_streaming_contract(component: Any, contract: type[Any]) -> bool:
        caps = capabilities_of(component)
        return (
            isinstance(component, contract)
            and caps.supports_streaming
            and not caps.requires_complete_sequence
        )
