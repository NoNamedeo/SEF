from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineExecutionPlan import PipelineExecutionPlan
from sef.core.pipeline.PipelineExecutionResult import PipelineExecutionResult
from sef.core.pipeline.PipelineRunOptions import (
    PipelineExecutionPlanLevel,
    PipelineRunOptions,
)
from sef.core.visualization.PipelineOutputs import PipelineOutputs
from sef.core.visualization.PipelineRunMetadata import PipelineRunMetadata


class PipelineOutputAssembler:
    """
    Builds the public ``PipelineOutputs`` object from raw execution data.

    The executor classes return domain data and artifacts only. This assembler
    owns metadata, execution-plan snapshots, and reproducibility exports so the
    execution path does not depend on serialization details.

    Boundary role
    -------------
    ``PipelineExecutionResult`` is an internal DTO. ``PipelineOutputs`` is the
    public contract consumed by scripts, tests and UI services. This assembler
    is the only component that should translate between those two shapes.
    """

    def __init__(
        self,
        *,
        context: PipelineContext,
        execution_plan: PipelineExecutionPlan | None,
        pipeline_id: str | None,
        execution_metadata: Mapping[str, Any],
        run_options: PipelineRunOptions,
    ) -> None:
        self._context = context
        self._execution_plan = execution_plan
        self._pipeline_id = pipeline_id
        self._execution_metadata = dict(execution_metadata)
        self._run_options = run_options

    def build(self, execution_result: PipelineExecutionResult) -> PipelineOutputs:
        """
        Return final outputs enriched with metadata and reproducibility data.

        The method is intentionally side-effect free: it derives metadata and
        export artifacts from the context, execution plan and raw result without
        mutating the context or the artifacts returned by executors.
        """
        outputs = PipelineOutputs(
            results=execution_result.results,
            final_artifacts=execution_result.final_artifacts,
            debug_artifacts=execution_result.debug_artifacts,
            metadata=PipelineRunMetadata(
                pipeline_id=self._resolved_pipeline_id(),
                generated_at=datetime.now(timezone.utc),
                execution_metadata={
                    **dict(self._execution_metadata),
                    "latency_policy_metrics": dict(execution_result.latency_policy_metrics),
                },
                execution_plan=self._execution_plan_metadata(),
            ),
            intermediate_frames=execution_result.intermediate_frames,
        )
        if not self._run_options.reproducibility:
            return outputs
        return self._attach_reproducibility(outputs)

    def _execution_plan_metadata(self) -> dict[str, Any]:
        if self._execution_plan is None:
            return {}
        if self._run_options.execution_plan is PipelineExecutionPlanLevel.SUMMARY:
            return self._execution_plan.as_summary_dict()
        if self._run_options.execution_plan is PipelineExecutionPlanLevel.FULL:
            return self._execution_plan.as_dict()
        return {}

    def _attach_reproducibility(self, outputs: PipelineOutputs) -> PipelineOutputs:
        from sef.core.pipeline.PipelineCodeExporter import PipelineCodeExporter
        from sef.core.pipeline.PipelineConfigExporter import PipelineConfigExporter

        config_exporter = PipelineConfigExporter()
        export_config = config_exporter.export(self._context, outputs, run_options=self._run_options)
        reproducibility = {
            "config": export_config,
            "json": config_exporter.to_json(export_config),
            "yaml": config_exporter.to_yaml(export_config),
            "python_builder_code": PipelineCodeExporter().export_config(export_config),
        }
        return PipelineOutputs(
            results=outputs.results,
            final_artifacts=outputs.final_artifacts,
            debug_artifacts=outputs.debug_artifacts,
            metadata=PipelineRunMetadata(
                pipeline_id=outputs.metadata.pipeline_id,
                generated_at=outputs.metadata.generated_at,
                execution_metadata=outputs.metadata.execution_metadata,
                execution_plan=outputs.metadata.execution_plan,
                reproducibility=reproducibility,
            ),
            intermediate_frames=outputs.intermediate_frames,
        )

    def _resolved_pipeline_id(self) -> str:
        return self._pipeline_id or "pipeline-unknown"
