from __future__ import annotations

from pprint import pformat
from textwrap import dedent
from typing import Any, Mapping

from library.core.pipeline.PipelineConfigExporter import PipelineConfigExporter
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.visualization.PipelineOutputs import PipelineOutputs


class PipelineCodeExporter:
    """
    Generate reproducible Python builder code for an executed pipeline.

    The generated code rebuilds through ConfigPipelineBuilder rather than
    importing concrete components directly. This keeps plugin resolution in the
    registry and mirrors the production configuration path.
    """

    def __init__(self, config_exporter: PipelineConfigExporter | None = None) -> None:
        self._config_exporter = config_exporter or PipelineConfigExporter()

    def export(
        self,
        context: PipelineContext,
        outputs: PipelineOutputs | None = None,
        export_config: Mapping[str, Any] | None = None,
    ) -> str:
        """Return executable Python source that rebuilds the pipeline context."""
        config = dict(export_config or self._config_exporter.export(context, outputs))
        return self.export_config(config)

    @staticmethod
    def export_config(export_config: Mapping[str, Any]) -> str:
        """Return executable Python source for an already-built export config."""
        config_literal = pformat(dict(export_config), width=100, sort_dicts=False)
        template = dedent(
            '''\
                from __future__ import annotations

                from typing import Any

                from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
                from library.core.pipeline.Pipeline import Pipeline
                from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
                from library.core.plugins.PluginRegistry import PluginRegistry, create_builtin_registry


                PIPELINE_EXPORT: dict[str, Any] = __PIPELINE_EXPORT__
                PIPELINE_CONFIG: dict[str, Any] = {
                    "schema_version": PIPELINE_EXPORT.get("schema_version", "1.0"),
                    "pipeline": PIPELINE_EXPORT["pipeline"],
                }


                def build_registry() -> PluginRegistry:
                    """
                    Build the registry used by this pipeline.

                    Register project-specific plugins here before calling build_context when the
                    export contains custom registered component names.
                    """
                    return create_builtin_registry()


                def build_context(registry: PluginRegistry | None = None):
                    """Rebuild the exported PipelineContext from its declarative config."""
                    resolved_registry = registry or build_registry()
                    return ConfigPipelineBuilder(resolved_registry).build_context(PIPELINE_CONFIG)


                def build_pipeline(
                    registry: PluginRegistry | None = None,
                    pipeline_id: str | None = None,
                    execution_metadata: dict[str, Any] | None = None,
                ) -> Pipeline:
                    """Create a Pipeline instance from the rebuilt context."""
                    return Pipeline(
                        build_context(registry),
                        pipeline_id=pipeline_id,
                        execution_metadata=execution_metadata,
                    )


                def run_pipeline(registry: PluginRegistry | None = None):
                    """Execute the rebuilt pipeline through the public orchestrator facade."""
                    execution = PIPELINE_EXPORT.get("execution", {})
                    return PipelineOrchestrator().run(
                        build_context(registry),
                        pipeline_id=execution.get("pipeline_id"),
                        execution_metadata=execution.get("metadata", {}),
                    )


                if __name__ == "__main__":
                    outputs = run_pipeline()
                    print(f"{len(outputs.results)} result(s), {outputs.artifact_count} artifact(s)")
                '''
        ).strip()
        return template.replace("__PIPELINE_EXPORT__", config_literal) + "\n"
