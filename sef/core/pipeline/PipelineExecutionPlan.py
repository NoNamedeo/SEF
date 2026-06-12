from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from library.core.interfaces.StageCapabilities import StageCapabilities


@dataclass(frozen=True, slots=True)
class ExecutionPlanStage:
    """
    Public description of one stage in an execution plan.

    The value is generated before execution starts and is safe to display in
    UIs, logs, or diagnostics. It contains decisions and rationale, not runtime
    executor objects.

    Attributes
    ----------
    stage_id:
        Stable stage identifier used in logs and errors.
    stage_group:
        Logical stage group, such as `frame_extraction`, `analysis`, or
        `visualisation`.
    component_name:
        Human-readable component class name.
    execution_mode:
        Selected mode, usually `batch` or `streaming`.
    capabilities:
        Capabilities declared by the component.
    materializes_input:
        Whether the stage creates a materialization boundary.
    reason:
        Policy explanation for the selected execution mode.
    """

    stage_id: str
    stage_group: str
    component_name: str
    execution_mode: str
    capabilities: StageCapabilities
    materializes_input: bool = False
    reason: str = ""
    estimated_queue_bytes: int | None = None
    estimated_materialized_bytes: int | None = None

    @property
    def streams(self) -> bool:
        """Return `True` when this stage is planned for streaming execution."""
        return self.execution_mode == "streaming"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe stage descriptor for APIs, UIs, and logs."""
        return {
            "stage_id": self.stage_id,
            "stage_group": self.stage_group,
            "component_name": self.component_name,
            "execution_mode": self.execution_mode,
            "capabilities": self.capabilities.as_dict(),
            "materializes_input": self.materializes_input,
            "reason": self.reason,
            "estimated_queue_bytes": self.estimated_queue_bytes,
            "estimated_materialized_bytes": self.estimated_materialized_bytes,
        }


@dataclass(frozen=True, slots=True)
class PipelineExecutionPlan:
    """
    Public execution plan generated before a pipeline run starts.

    The plan is an observability contract. It explains which stages are
    streamable, where the runtime will materialize data, and why each decision
    was selected by the execution policy.
    """

    stages: tuple[ExecutionPlanStage, ...]
    runtime: dict[str, Any] = field(default_factory=dict)

    @property
    def streamable_end_to_end(self) -> bool:
        """Return whether all stages preserve streaming without materialization."""
        return all(stage.streams and not stage.materializes_input for stage in self.stages)

    @property
    def materialization_boundaries(self) -> tuple[ExecutionPlanStage, ...]:
        """Return stages that explicitly materialize an upstream stream."""
        return tuple(stage for stage in self.stages if stage.materializes_input)

    def by_group(self, group: str) -> tuple[ExecutionPlanStage, ...]:
        """
        Return stages belonging to a logical group.

        Parameters
        ----------
        group:
            Stage group identifier.
        """
        return tuple(stage for stage in self.stages if stage.stage_group == group)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe execution plan for metadata and API responses."""
        return {
            "streamable_end_to_end": self.streamable_end_to_end,
            "runtime": dict(self.runtime),
            "materialization_boundaries": [
                {
                    "stage_id": stage.stage_id,
                    "component_name": stage.component_name,
                    "estimated_materialized_bytes": stage.estimated_materialized_bytes,
                }
                for stage in self.materialization_boundaries
            ],
            "stages": [stage.as_dict() for stage in self.stages],
        }

    def as_text(self) -> str:
        """Return a stable, human-readable execution plan summary."""
        lines = [
            "Pipeline execution plan:",
            f"- streamable_end_to_end={self.streamable_end_to_end}",
            f"- latency_policy={self.runtime.get('latency_policy', {}).get('name', 'unknown')}",
        ]
        for stage in self.stages:
            boundary = " materializes_input" if stage.materializes_input else ""
            lines.append(
                f"- {stage.stage_id}: {stage.component_name} [{stage.execution_mode}]{boundary}"
                + (f" - {stage.reason}" if stage.reason else "")
            )
        return "\n".join(lines)


def capabilities_of(component: Any) -> StageCapabilities:
    capabilities = getattr(component, "capabilities", None)
    if isinstance(capabilities, StageCapabilities):
        return capabilities
    return StageCapabilities.batch()


def stages_by_group(stages: Iterable[ExecutionPlanStage]) -> dict[str, tuple[ExecutionPlanStage, ...]]:
    grouped: dict[str, list[ExecutionPlanStage]] = {}
    for stage in stages:
        grouped.setdefault(stage.stage_group, []).append(stage)
    return {key: tuple(value) for key, value in grouped.items()}
