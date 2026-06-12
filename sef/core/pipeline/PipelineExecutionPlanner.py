from __future__ import annotations

from typing import Any

from sef.core.pipeline.PipelineComponentCapabilities import PipelineComponentCapabilities
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineExecutionLookahead import PipelineExecutionLookahead
from sef.core.pipeline.PipelineExecutionPolicy import (
    DefaultPipelineExecutionPolicy,
    PipelineExecutionEstimates,
    PipelineExecutionPolicy,
    PipelineStagePolicyContext,
)
from sef.core.pipeline.PipelineExecutionPlan import (
    ExecutionPlanStage,
    PipelineExecutionPlan,
    capabilities_of,
)
from sef.core.pipeline.VisualizerBinding import VisualizerBinding


class PipelineExecutionPlanner:
    """
    Builds the execution plan used to explain adaptive runtime decisions.

    The planner is read-only: it inspects the context and component
    capabilities, but it never executes stages or mutates components. Its output
    is attached to ``PipelineOutputs`` so users can understand why a run used
    streaming, batch execution, or materialization boundaries.
    """

    def __init__(self, execution_policy: PipelineExecutionPolicy | None = None) -> None:
        self._execution_policy = execution_policy or DefaultPipelineExecutionPolicy()

    def build(self, context: PipelineContext) -> PipelineExecutionPlan:
        """
        Build a plan for the provided context.

        The returned plan mirrors the same capability and policy rules used by
        ``SegmentedPipelineExecutor``. Keeping both paths aligned is essential:
        what the user sees before execution must match what the runtime does.
        """
        estimates = PipelineExecutionEstimates.from_context(context)
        lookahead = PipelineExecutionLookahead(context)
        stages: list[ExecutionPlanStage] = []

        frame_decision = self._execution_policy.decide_source(
            PipelineStagePolicyContext(
                stage_id="frame_extraction",
                stage_group="frame_extractor",
                stage_streamable=self._is_streaming_frame_extractor(context.frame_extractor),
                downstream_streamable=lookahead.frame_successor_streamable(processor_index=0),
                estimated_queue_bytes=estimates.frame_queue_bytes,
                estimated_materialized_bytes=estimates.materialized_frame_bytes,
            )
        )
        frame_stream_pending = frame_decision.streams
        stages.append(
            self._stage(
                "frame_extraction",
                "frame_extractor",
                context.frame_extractor,
                streaming=frame_stream_pending,
                reason=frame_decision.reason,
                estimated_queue_bytes=estimates.frame_queue_bytes if frame_stream_pending else None,
            )
        )

        for index, processor in enumerate(context.frame_processors):
            decision = self._execution_policy.decide_stage(
                PipelineStagePolicyContext(
                    stage_id=f"frame_processing[{index}]",
                    stage_group="frame_processors",
                    stage_streamable=self._is_streaming_frame_processor(processor),
                    input_is_streaming=frame_stream_pending,
                    downstream_streamable=lookahead.frame_successor_streamable(processor_index=index + 1),
                    estimated_queue_bytes=estimates.frame_queue_bytes,
                    estimated_materialized_bytes=estimates.materialized_frame_bytes,
                )
            )
            streaming = decision.streams
            materializes = frame_stream_pending and not streaming
            stages.append(
                self._stage(
                    f"frame_processing[{index}]",
                    "frame_processors",
                    processor,
                    streaming=streaming,
                    materializes_input=materializes,
                    reason=decision.reason,
                    estimated_queue_bytes=estimates.frame_queue_bytes if streaming else None,
                    estimated_materialized_bytes=estimates.materialized_frame_bytes if materializes else None,
                )
            )
            frame_stream_pending = streaming

        for index, exporter in enumerate(context.frame_exporters):
            decision = self._execution_policy.decide_stage(
                PipelineStagePolicyContext(
                    stage_id=f"frame_export[{index}]",
                    stage_group="frame_exporters",
                    stage_streamable=self._is_streaming_frame_exporter(exporter),
                    input_is_streaming=frame_stream_pending,
                    downstream_streamable=lookahead.frame_export_successor_streamable(exporter_index=index + 1),
                    estimated_queue_bytes=estimates.frame_queue_bytes,
                    estimated_materialized_bytes=estimates.materialized_frame_bytes,
                )
            )
            streaming = decision.streams
            materializes = frame_stream_pending and not streaming
            stages.append(
                self._stage(
                    f"frame_export[{index}]",
                    "frame_exporters",
                    exporter,
                    streaming=streaming,
                    materializes_input=materializes,
                    reason=decision.reason,
                    estimated_queue_bytes=estimates.frame_queue_bytes if streaming else None,
                    estimated_materialized_bytes=estimates.materialized_frame_bytes if materializes else None,
                )
            )
            frame_stream_pending = streaming

        signal_decision = self._execution_policy.decide_stage(
            PipelineStagePolicyContext(
                stage_id="signal_extraction",
                stage_group="signal_extractor",
                stage_streamable=self._is_streaming_signal_extractor(context.signal_extractor),
                input_is_streaming=frame_stream_pending,
                downstream_streamable=lookahead.signal_successor_streamable(cleaner_index=0),
                estimated_queue_bytes=estimates.signal_queue_bytes,
                estimated_materialized_bytes=estimates.materialized_frame_bytes,
            )
        )
        signal_streaming = signal_decision.streams
        signal_materializes = frame_stream_pending and not signal_streaming
        stages.append(
            self._stage(
                "signal_extraction",
                "signal_extractor",
                context.signal_extractor,
                streaming=signal_streaming,
                materializes_input=signal_materializes,
                reason=signal_decision.reason,
                estimated_queue_bytes=estimates.signal_queue_bytes if signal_streaming else None,
                estimated_materialized_bytes=estimates.materialized_frame_bytes if signal_materializes else None,
            )
        )

        for index, cleaner in enumerate(context.signal_cleaners):
            decision = self._execution_policy.decide_stage(
                PipelineStagePolicyContext(
                    stage_id=f"signal_cleaning[{index}]",
                    stage_group="signal_cleaners",
                    stage_streamable=self._is_streaming_signal_cleaner(cleaner),
                    input_is_streaming=signal_streaming,
                    downstream_streamable=lookahead.signal_successor_streamable(cleaner_index=index + 1),
                    estimated_queue_bytes=estimates.signal_queue_bytes,
                )
            )
            streaming = decision.streams
            materializes = signal_streaming and not streaming
            stages.append(
                self._stage(
                    f"signal_cleaning[{index}]",
                    "signal_cleaners",
                    cleaner,
                    streaming=streaming,
                    materializes_input=materializes,
                    reason=decision.reason,
                    estimated_queue_bytes=estimates.signal_queue_bytes if streaming else None,
                )
            )
            signal_streaming = streaming

        analyzer_streaming: list[bool] = []
        streaming_visualizer_result_indexes = self._streaming_visualizer_result_indexes(context)
        for index, analyzer in enumerate(context.analyzers):
            decision = self._execution_policy.decide_analyzer(
                PipelineStagePolicyContext(
                    stage_id=f"analysis[{index}]",
                    stage_group="analyzers",
                    stage_streamable=self._is_streaming_analyzer(analyzer),
                    input_is_streaming=signal_streaming,
                    progressive_consumer=index in streaming_visualizer_result_indexes,
                    estimated_queue_bytes=estimates.data_queue_bytes,
                )
            )
            streaming = decision.streams
            analyzer_streaming.append(streaming)
            stages.append(
                self._stage(
                    f"analysis[{index}]",
                    "analyzers",
                    analyzer,
                    streaming=streaming,
                    materializes_input=signal_streaming and not streaming,
                    reason=decision.reason,
                    estimated_queue_bytes=estimates.data_queue_bytes if streaming else None,
                )
            )

        for index, binding in enumerate(self._visualizer_bindings(context)):
            target_indexes = binding.target_indexes(len(context.analyzers))
            streaming = PipelineComponentCapabilities.can_stream_visualizer(binding.visualizer) and any(
                analyzer_streaming[target_index] for target_index in target_indexes
            )
            stages.append(
                self._stage(
                    f"visualisation[{index}]",
                    "visualizers",
                    binding.visualizer,
                    streaming=streaming,
                    reason="consumes progressive analyzer data" if streaming else "renders final analyzer result",
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
        return PipelineComponentCapabilities.can_stream_frame_extractor(component)

    @staticmethod
    def _is_streaming_frame_processor(component: Any) -> bool:
        return PipelineComponentCapabilities.can_stream_frame_processor(component)

    @staticmethod
    def _is_streaming_frame_exporter(component: Any) -> bool:
        return PipelineComponentCapabilities.can_stream_frame_exporter(component)

    @staticmethod
    def _is_streaming_signal_extractor(component: Any) -> bool:
        return PipelineComponentCapabilities.can_stream_signal_extractor(component)

    @staticmethod
    def _is_streaming_signal_cleaner(component: Any) -> bool:
        return PipelineComponentCapabilities.can_stream_signal_cleaner(component)

    @staticmethod
    def _is_streaming_analyzer(component: Any) -> bool:
        return PipelineComponentCapabilities.can_stream_analyzer(component)

    def _streaming_visualizer_result_indexes(self, context: PipelineContext) -> set[int]:
        result_indexes: set[int] = set()
        for binding in self._visualizer_bindings(context):
            if not PipelineComponentCapabilities.can_stream_visualizer(binding.visualizer):
                continue
            result_indexes.update(binding.target_indexes(len(context.analyzers)))
        return result_indexes

    @staticmethod
    def _visualizer_bindings(context: PipelineContext) -> list[VisualizerBinding]:
        return [
            *(VisualizerBinding(visualizer) for visualizer in context.visualizers),
            *context.visualizer_bindings,
        ]
