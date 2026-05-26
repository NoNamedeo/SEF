"""Stable public contracts for the SEF core package.

`library.core` is the preferred import surface for framework users who need the
standard pipeline facade, declarative builder, typed errors, registry contracts,
selected artifacts, visualization outputs, and realtime preview handoff values.

The package intentionally re-exports only contracts that are stable enough for
external integrations. Execution collaborators, planner internals, and concrete
OpenCV/YOLO/Streamlit adapters live behind narrower package exports or outside
the core layer.

Examples
--------
```python
from library.core import ConfigPipelineBuilder, Pipeline, PluginRegistry

registry = PluginRegistry()
context = ConfigPipelineBuilder(registry).build_context(config)
outputs = Pipeline(context).run()
```
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ConfigPipelineBuilder": ("library.core.pipeline.ConfigPipelineBuilder", "ConfigPipelineBuilder"),
    "ConfigSchemaError": ("library.core.errors", "ConfigSchemaError"),
    "ConfigVersionError": ("library.core.errors", "ConfigVersionError"),
    "CURRENT_PIPELINE_CONFIG_VERSION": (
        "library.core.pipeline.PipelineConfigVersioning",
        "CURRENT_PIPELINE_CONFIG_VERSION",
    ),
    "DuplicatePluginRegistrationError": ("library.core.errors", "DuplicatePluginRegistrationError"),
    "ExecutionPlanStage": ("library.core.pipeline.PipelineExecutionPlan", "ExecutionPlanStage"),
    "FluentPipelineBuilder": ("library.core.pipeline.FluentPipelineBuilder", "FluentPipelineBuilder"),
    "Frame": ("library.core.artifacts.Frame", "Frame"),
    "FrameBuffer": ("library.core.artifacts.FrameBuffer", "FrameBuffer"),
    "IRealtimeFrameSink": ("library.core.realtime.IRealtimeFrameSink", "IRealtimeFrameSink"),
    "InvalidPluginRegistrationError": ("library.core.errors", "InvalidPluginRegistrationError"),
    "LatencyPolicyConfig": ("library.core.pipeline.LatencyPolicy", "LatencyPolicyConfig"),
    "PIPELINE_CONFIG_VERSION_KEY": (
        "library.core.pipeline.PipelineConfigVersioning",
        "PIPELINE_CONFIG_VERSION_KEY",
    ),
    "Pipeline": ("library.core.pipeline.Pipeline", "Pipeline"),
    "PipelineConfigurationError": ("library.core.errors", "PipelineConfigurationError"),
    "PipelineContext": ("library.core.pipeline.PipelineContext", "PipelineContext"),
    "PipelineContextError": ("library.core.errors", "PipelineContextError"),
    "PipelineError": ("library.core.errors", "PipelineError"),
    "PipelineExecutionError": ("library.core.errors", "PipelineExecutionError"),
    "PipelineExecutionPlan": ("library.core.pipeline.PipelineExecutionPlan", "PipelineExecutionPlan"),
    "PipelineOutputs": ("library.core.visualization.PipelineOutputs", "PipelineOutputs"),
    "PipelineRunAlreadyActiveError": ("library.core.errors", "PipelineRunAlreadyActiveError"),
    "PluginCategory": ("library.core.plugins", "PluginCategory"),
    "PluginConstructionError": ("library.core.errors", "PluginConstructionError"),
    "PluginDefinition": ("library.core.plugins", "PluginDefinition"),
    "PluginRegistryError": ("library.core.errors", "PluginRegistryError"),
    "PluginRegistry": ("library.core.plugins", "PluginRegistry"),
    "PluginResolutionError": ("library.core.errors", "PluginResolutionError"),
    "RealtimeFrame": ("library.core.realtime.RealtimeFrame", "RealtimeFrame"),
    "SEFError": ("library.core.errors", "SEFError"),
    "StageCapabilities": ("library.core.interfaces.StageCapabilities", "StageCapabilities"),
    "StageErrorContext": ("library.core.errors", "StageErrorContext"),
    "StageExecutionError": ("library.core.errors", "StageExecutionError"),
    "StreamAbortedError": ("library.core.errors", "StreamAbortedError"),
    "StreamRuntimeError": ("library.core.errors", "StreamRuntimeError"),
    "StreamRuntimeConfig": ("library.core.pipeline.StreamRuntimeConfig", "StreamRuntimeConfig"),
    "TextArtifact": ("library.core.visualization.VisualArtifact", "TextArtifact"),
    "VisualArtifact": ("library.core.visualization.VisualArtifact", "VisualArtifact"),
    "VisualizationContext": ("library.core.visualization.VisualizationContext", "VisualizationContext"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
