from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from sef.core.interfaces.pipeline.IEventBus import IEventBus
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineErrors import PipelineExecutionError
from sef.core.pipeline.PipelineEventInjector import PipelineEventInjector
from sef.core.pipeline.PipelineExecutionPlan import PipelineExecutionPlan
from sef.core.pipeline.PipelineExecutionPlanner import PipelineExecutionPlanner
from sef.core.pipeline.PipelineExecutionPolicy import (
    DefaultPipelineExecutionPolicy,
    PipelineExecutionPolicy,
)
from sef.core.pipeline.PipelineOutputAssembler import PipelineOutputAssembler
from sef.core.pipeline.PipelineStageExecutor import PipelineStageExecutor
from sef.core.pipeline.SegmentedPipelineExecutor import SegmentedPipelineExecutor
from sef.core.pipeline.VisualizationExecutor import VisualizationExecutor
from sef.core.visualization.PipelineOutputs import PipelineOutputs

__all__ = ["Pipeline", "PipelineExecutionError"]

log = logging.getLogger(__name__)


class Pipeline:
    """
    Public execution facade for a single pipeline run.

    ``Pipeline`` intentionally coordinates collaborators instead of owning the
    workflow details. Frame execution, signal analysis, visualization, event
    injection, and output assembly are delegated to focused collaborators in
    this package. This keeps the public API stable while avoiding a god object.

    Responsibilities
    ----------------
    - Validate no business rule directly; validation belongs to
      ``PipelineContext`` and the concrete components.
    - Build a stable execution plan before the run starts.
    - Keep execution-mode decisions behind ``PipelineExecutionPolicy``.
    - Inject runtime event metadata into event-aware components.
    - Delegate execution to ``SegmentedPipelineExecutor``.
    - Convert the internal execution result into public ``PipelineOutputs``.

    Extension notes
    ---------------
    New execution behavior should normally be introduced through a custom
    ``PipelineExecutionPolicy`` or through segmented runtime collaborators
    rather than by adding stage logic here. This class should remain thin so
    callers can treat it as a stable facade.
    """

    def __init__(
        self,
        context: PipelineContext,
        event_bus: IEventBus | None = None,
        pipeline_id: str | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
        execution_policy: PipelineExecutionPolicy | None = None,
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
        execution_policy:
            Optional strategy used by planner and runtime to choose batch or
            streaming execution for each stage.
        """
        self._context = context
        self._event_bus = event_bus
        self._pipeline_id = pipeline_id
        self._execution_metadata = dict(execution_metadata or {})
        self._execution_policy = execution_policy or DefaultPipelineExecutionPolicy()
        self._execution_plan = PipelineExecutionPlanner(self._execution_policy).build(context)
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

    def _build_executor(self) -> SegmentedPipelineExecutor:
        visualization_executor = VisualizationExecutor(
            context=self._context,
            stage_executor=self._stage_executor,
            pipeline_id=self._pipeline_id,
            execution_metadata=self._execution_metadata,
        )
        return SegmentedPipelineExecutor(
            context=self._context,
            stage_executor=self._stage_executor,
            visualization_executor=visualization_executor,
            execution_policy=self._execution_policy,
            pipeline_id=self._pipeline_id,
            execution_metadata=self._execution_metadata,
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
