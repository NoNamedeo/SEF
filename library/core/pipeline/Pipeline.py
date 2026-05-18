from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.pipeline.AdaptivePipelineExecutor import AdaptivePipelineExecutor
from library.core.pipeline.FrameExporterExecutor import FrameExporterExecutor
from library.core.pipeline.FramePipelineExecutor import FramePipelineExecutor
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineErrors import PipelineExecutionError
from library.core.pipeline.PipelineEventInjector import PipelineEventInjector
from library.core.pipeline.PipelineExecutionPlan import PipelineExecutionPlan
from library.core.pipeline.PipelineExecutionPlanner import PipelineExecutionPlanner
from library.core.pipeline.PipelineOutputAssembler import PipelineOutputAssembler
from library.core.pipeline.PipelineStageExecutor import PipelineStageExecutor
from library.core.pipeline.SignalPipelineExecutor import SignalPipelineExecutor
from library.core.pipeline.StreamingSignalTailExecutor import StreamingSignalTailExecutor
from library.core.pipeline.VisualizationExecutor import VisualizationExecutor
from library.core.visualization.PipelineOutputs import PipelineOutputs

__all__ = ["Pipeline", "PipelineExecutionError"]

log = logging.getLogger(__name__)


class Pipeline:
    """
    Public execution facade for a single pipeline run.

    ``Pipeline`` intentionally coordinates collaborators instead of owning the
    workflow details. Frame execution, signal analysis, visualization, event
    injection, and output assembly are delegated to focused components in this
    package. This keeps the public API stable while avoiding a god object.

    Responsibilities
    ----------------
    - Validate no business rule directly; validation belongs to
      ``PipelineContext`` and the concrete components.
    - Build a stable execution plan before the run starts.
    - Inject runtime event metadata into event-aware components.
    - Delegate execution to ``AdaptivePipelineExecutor``.
    - Convert the internal execution result into public ``PipelineOutputs``.

    Extension notes
    ---------------
    New execution behavior should normally be introduced through one of the
    focused executors rather than by adding stage logic here. This class should
    remain thin so callers can treat it as a stable facade.
    """

    def __init__(
        self,
        context: PipelineContext,
        event_bus: IEventBus | None = None,
        pipeline_id: str | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Create a pipeline facade for an already-built context.

        Parameters
        ----------
        context:
            Immutable set of pipeline components and runtime configuration.
        event_bus:
            Optional domain event bus injected into components implementing
            ``IEventEmitter``.
        pipeline_id:
            Optional stable identifier propagated into metadata and artifacts.
        execution_metadata:
            Additional metadata copied into event contexts, visualizer contexts,
            exporter contexts, and final run metadata.
        """
        self._context = context
        self._event_bus = event_bus
        self._pipeline_id = pipeline_id
        self._execution_metadata = dict(execution_metadata or {})
        self._execution_plan = PipelineExecutionPlanner().build(context)
        self._stage_executor = PipelineStageExecutor()

    def run(self) -> PipelineOutputs:
        """
        Execute the pipeline and return analyzer results plus artifacts.

        Returns
        -------
        PipelineOutputs
            Final analyzer results, visual artifacts, debug artifacts,
            execution metadata, execution plan, and reproducibility exports.

        Raises
        ------
        PipelineExecutionError
            If any stage delegated to a component fails.
        """
        runtime_metadata = self._runtime_metadata()
        PipelineEventInjector().inject(
            context=self._context,
            event_bus=self._event_bus,
            metadata=runtime_metadata,
        )
        log.info("%s", self._execution_plan.as_text())
        execution_result = self._build_executor().run()
        return self._build_output_assembler().build(execution_result)

    def execution_plan(self) -> PipelineExecutionPlan:
        """
        Return the adaptive execution plan that will be used by ``run``.

        The plan is computed once during construction so callers can inspect
        streaming decisions and materialization boundaries before execution.
        """
        return self._execution_plan

    def _build_executor(self) -> AdaptivePipelineExecutor:
        frame_exporter_executor = FrameExporterExecutor(
            context=self._context,
            stage_executor=self._stage_executor,
            pipeline_id=self._pipeline_id,
            execution_metadata=self._execution_metadata,
        )
        visualization_executor = VisualizationExecutor(
            context=self._context,
            stage_executor=self._stage_executor,
            pipeline_id=self._pipeline_id,
            execution_metadata=self._execution_metadata,
        )
        return AdaptivePipelineExecutor(
            context=self._context,
            frame_pipeline_executor=FramePipelineExecutor(
                context=self._context,
                stage_executor=self._stage_executor,
            ),
            frame_exporter_executor=frame_exporter_executor,
            signal_pipeline_executor=SignalPipelineExecutor(
                context=self._context,
                stage_executor=self._stage_executor,
            ),
            streaming_tail_executor=StreamingSignalTailExecutor(
                context=self._context,
                stage_executor=self._stage_executor,
                frame_exporter_executor=frame_exporter_executor,
                visualization_executor=visualization_executor,
            ),
            visualization_executor=visualization_executor,
        )

    def _build_output_assembler(self) -> PipelineOutputAssembler:
        return PipelineOutputAssembler(
            context=self._context,
            execution_plan=self._execution_plan,
            pipeline_id=self._pipeline_id,
            execution_metadata=self._execution_metadata,
        )

    def _runtime_metadata(self) -> dict[str, Any]:
        metadata = dict(self._execution_metadata)
        if self._pipeline_id is not None:
            metadata.setdefault("pipeline_id", self._pipeline_id)
        return metadata
