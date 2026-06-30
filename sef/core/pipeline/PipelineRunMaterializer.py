from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sef.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineErrors import PipelineConfigurationError
from sef.core.pipeline.PipelineRunOptions import PipelineRunOptions
from sef.core.plugins.PluginRegistry import PluginRegistry


@dataclass(frozen=True, slots=True)
class MaterializedPipelineRun:
    """
    Runtime-ready view of a declarative run config.

    The public API works with run documents. Execution still needs a validated
    ``PipelineContext`` and typed options, so this DTO marks the application
    boundary between declaration and runtime objects.
    """

    context: PipelineContext
    pipeline_id: str | None
    execution_metadata: Mapping[str, Any]
    run_options: PipelineRunOptions


class PipelineRunMaterializer:
    """
    Convert a run config into the immutable runtime context used by Pipeline.

    This collaborator keeps config parsing and plugin construction out of the
    orchestrator. The orchestrator remains responsible for coordinating the
    runner/factory/buses; this class owns only materialization.
    """

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        self._registry = registry

    def materialize(
        self,
        config: Mapping[str, Any],
    ) -> MaterializedPipelineRun:
        """Build a ``PipelineContext`` and resolve run-level execution settings."""
        if self._registry is None:
            raise PipelineConfigurationError(
                "A PluginRegistry is required to materialize a run config. "
                "Pass a registry to PipelineOrchestrator or use the high-level sef.run API."
            )

        context = ConfigPipelineBuilder(self._registry).build_context(config)
        source_config = context.source_config
        return MaterializedPipelineRun(
            context=context,
            pipeline_id=_config_pipeline_id(source_config),
            execution_metadata=_config_metadata(source_config),
            run_options=PipelineRunOptions.from_config(source_config),
        )


def _config_pipeline_id(config: Mapping[str, Any]) -> str | None:
    value = config.get("id")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PipelineConfigurationError("Run config field 'id' must be a non-empty string.", path="id")
    return value.strip()


def _config_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("metadata", {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PipelineConfigurationError("Run config field 'metadata' must be a mapping.", path="metadata")
    return dict(value)


__all__ = ["MaterializedPipelineRun", "PipelineRunMaterializer"]
