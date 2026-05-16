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
from library.core.pipeline.PipelineExecutionPlan import (
    ExecutionPlanStage,
    PipelineExecutionPlan,
    capabilities_of,
)


class PipelineExecutionPlanner:
    """Build the single source of truth used to explain adaptive execution."""

    def build(self, context: PipelineContext) -> PipelineExecutionPlan:
        frame_bytes = self._estimated_frame_bytes(context)
        queue_bytes = self._queue_bytes(frame_bytes, context.stream_runtime.frame_buffer_size)
        materialized_bytes = self._materialized_bytes(frame_bytes, context)
        stages: list[ExecutionPlanStage] = []

        frame_stream_pending = self._is_streaming_frame_extractor(context.frame_extractor)
        stages.append(
            self._stage(
                "frame_extraction",
                "frame_extractor",
                context.frame_extractor,
                streaming=frame_stream_pending,
                reason="explicit streaming frame extractor" if frame_stream_pending else "batch frame extractor",
                estimated_queue_bytes=queue_bytes if frame_stream_pending else None,
            )
        )

        for index, processor in enumerate(context.frame_processors):
            streaming = self._is_streaming_frame_processor(processor)
            materializes = frame_stream_pending and not streaming
            stages.append(
                self._stage(
                    f"frame_processing[{index}]",
                    "frame_processors",
                    processor,
                    streaming=streaming,
                    materializes_input=materializes,
                    reason="explicit streaming frame processor" if streaming else "requires complete frame sequence",
                    estimated_queue_bytes=queue_bytes if streaming else None,
                    estimated_materialized_bytes=materialized_bytes if materializes else None,
                )
            )
            frame_stream_pending = streaming

        exporters_streaming = all(self._is_streaming_frame_exporter(exporter) for exporter in context.frame_exporters)
        signal_tail_streaming = self._is_streaming_signal_tail(context)
        for index, exporter in enumerate(context.frame_exporters):
            streaming = exporters_streaming and signal_tail_streaming and self._is_streaming_frame_exporter(exporter)
            materializes = frame_stream_pending and not streaming
            stages.append(
                self._stage(
                    f"frame_export[{index}]",
                    "frame_exporters",
                    exporter,
                    streaming=streaming,
                    materializes_input=materializes,
                    reason="streaming frame exporter" if streaming else "frame tail materializes before export",
                    estimated_queue_bytes=queue_bytes if streaming else None,
                    estimated_materialized_bytes=materialized_bytes if materializes else None,
                )
            )
            frame_stream_pending = streaming

        signal_streaming = signal_tail_streaming and self._is_streaming_signal_extractor(context.signal_extractor)
        signal_materializes = frame_stream_pending and not signal_streaming
        stages.append(
            self._stage(
                "signal_extraction",
                "signal_extractor",
                context.signal_extractor,
                streaming=signal_streaming,
                materializes_input=signal_materializes,
                reason="explicit streaming signal extractor" if signal_streaming else "batch signal extractor or batch downstream tail",
                estimated_queue_bytes=self._signal_queue_bytes(context) if signal_streaming else None,
                estimated_materialized_bytes=materialized_bytes if signal_materializes else None,
            )
        )

        for index, cleaner in enumerate(context.signal_cleaners):
            streaming = signal_tail_streaming and self._is_streaming_signal_cleaner(cleaner)
            stages.append(
                self._stage(
                    f"signal_cleaning[{index}]",
                    "signal_cleaners",
                    cleaner,
                    streaming=streaming,
                    reason="explicit streaming signal cleaner" if streaming else "batch signal cleaner",
                    estimated_queue_bytes=self._signal_queue_bytes(context) if streaming else None,
                )
            )

        for index, analyzer in enumerate(context.analyzers):
            streaming = signal_tail_streaming and self._is_streaming_analyzer(analyzer)
            stages.append(
                self._stage(
                    f"analysis[{index}]",
                    "analyzers",
                    analyzer,
                    streaming=streaming,
                    reason="explicit streaming analyzer" if streaming else "batch analyzer",
                    estimated_queue_bytes=self._data_queue_bytes(context) if streaming else None,
                )
            )

        visualizers = [*context.visualizers, *(binding.visualizer for binding in context.visualizer_bindings)]
        for index, visualizer in enumerate(visualizers):
            streaming = signal_tail_streaming and isinstance(visualizer, IStreamingVisualizer)
            stages.append(
                self._stage(
                    f"visualisation[{index}]",
                    "visualizers",
                    visualizer,
                    streaming=streaming,
                    reason="explicit streaming visualizer" if streaming else "final artifact visualizer",
                )
            )

        return PipelineExecutionPlan(
            stages=tuple(stages),
            runtime=context.stream_runtime.as_dict(),
        )

    @staticmethod
    def _stage(
        stage_id: str,
        stage_group: str,
        component: Any,
        *,
        streaming: bool,
        reason: str,
        materializes_input: bool = False,
        estimated_queue_bytes: int | None = None,
        estimated_materialized_bytes: int | None = None,
    ) -> ExecutionPlanStage:
        return ExecutionPlanStage(
            stage_id=stage_id,
            stage_group=stage_group,
            component_name=type(component).__name__,
            execution_mode="streaming" if streaming else "batch",
            capabilities=capabilities_of(component),
            materializes_input=materializes_input,
            reason=reason,
            estimated_queue_bytes=estimated_queue_bytes,
            estimated_materialized_bytes=estimated_materialized_bytes,
        )

    @staticmethod
    def _is_streaming_frame_extractor(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingFrameExtractor) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def _is_streaming_frame_processor(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingFrameBufferProcessor) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def _is_streaming_frame_exporter(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingFrameExporter) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def _is_streaming_signal_extractor(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingSignalExtractor) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def _is_streaming_signal_cleaner(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingSignalCleaner) and caps.supports_streaming and not caps.requires_complete_sequence

    @staticmethod
    def _is_streaming_analyzer(component: Any) -> bool:
        caps = capabilities_of(component)
        return isinstance(component, IStreamingAnalyzer) and caps.supports_streaming and not caps.requires_complete_sequence

    def _is_streaming_signal_tail(self, context: PipelineContext) -> bool:
        visualizers = [*context.visualizers, *(binding.visualizer for binding in context.visualizer_bindings)]
        return (
            self._is_streaming_signal_extractor(context.signal_extractor)
            and all(self._is_streaming_signal_cleaner(cleaner) for cleaner in context.signal_cleaners)
            and all(self._is_streaming_analyzer(analyzer) for analyzer in context.analyzers)
            and all(isinstance(visualizer, IStreamingVisualizer) for visualizer in visualizers)
        )

    @staticmethod
    def _estimated_frame_bytes(context: PipelineContext) -> int | None:
        resize = getattr(context.frame_extractor, "resize", None)
        if not isinstance(resize, (tuple, list)) or len(resize) != 2:
            return None
        width, height = int(resize[0]), int(resize[1])
        if width <= 0 or height <= 0:
            return None
        return width * height * 3

    @staticmethod
    def _queue_bytes(frame_bytes: int | None, capacity: int) -> int | None:
        if frame_bytes is None:
            return None
        return frame_bytes * capacity

    @staticmethod
    def _signal_queue_bytes(context: PipelineContext) -> int:
        return context.stream_runtime.signal_buffer_size * 1024

    @staticmethod
    def _data_queue_bytes(context: PipelineContext) -> int:
        return context.stream_runtime.data_buffer_size * 1024

    @staticmethod
    def _materialized_bytes(frame_bytes: int | None, context: PipelineContext) -> int | None:
        if frame_bytes is None:
            return None
        max_frames = getattr(context.frame_extractor, "max_frames", None)
        if max_frames is None:
            return None
        return frame_bytes * int(max_frames)
