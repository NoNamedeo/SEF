from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AdaptiveSamplingFrameLatencyPolicy": ("library.core.pipeline.LatencyPolicy", "AdaptiveSamplingFrameLatencyPolicy"),
    "BlockingFrameLatencyPolicy": ("library.core.pipeline.LatencyPolicy", "BlockingFrameLatencyPolicy"),
    "ConfigPipelineBuilder": ("library.core.pipeline.ConfigPipelineBuilder", "ConfigPipelineBuilder"),
    "ConfigSchemaError": ("library.core.errors", "ConfigSchemaError"),
    "DefaultPipelineExecutionPolicy": ("library.core.pipeline.PipelineExecutionPolicy", "DefaultPipelineExecutionPolicy"),
    "DropNewestFrameLatencyPolicy": ("library.core.pipeline.LatencyPolicy", "DropNewestFrameLatencyPolicy"),
    "DropOldestFrameLatencyPolicy": ("library.core.pipeline.LatencyPolicy", "DropOldestFrameLatencyPolicy"),
    "ExecutionPlanStage": ("library.core.pipeline.PipelineExecutionPlan", "ExecutionPlanStage"),
    "FluentPipelineBuilder": ("library.core.pipeline.FluentPipelineBuilder", "FluentPipelineBuilder"),
    "FrameLatencyPolicy": ("library.core.pipeline.LatencyPolicy", "FrameLatencyPolicy"),
    "LatencyPolicyConfig": ("library.core.pipeline.LatencyPolicy", "LatencyPolicyConfig"),
    "Pipeline": ("library.core.pipeline.Pipeline", "Pipeline"),
    "PipelineConfigurationError": ("library.core.errors", "PipelineConfigurationError"),
    "PipelineContext": ("library.core.pipeline.PipelineContext", "PipelineContext"),
    "PipelineContextError": ("library.core.errors", "PipelineContextError"),
    "PipelineError": ("library.core.errors", "PipelineError"),
    "PipelineExecutionDecision": ("library.core.pipeline.PipelineExecutionPolicy", "PipelineExecutionDecision"),
    "PipelineExecutionError": ("library.core.errors", "PipelineExecutionError"),
    "PipelineExecutionMode": ("library.core.pipeline.PipelineExecutionPolicy", "PipelineExecutionMode"),
    "PipelineExecutionPlan": ("library.core.pipeline.PipelineExecutionPlan", "PipelineExecutionPlan"),
    "PipelineOrchestrator": ("library.core.pipeline.PipelineOrchestrator", "PipelineOrchestrator"),
    "PipelineRunSnapshot": ("library.core.pipeline.PipelineRunSnapshot", "PipelineRunSnapshot"),
    "PipelineRunState": ("library.core.pipeline.PipelineRunSnapshot", "PipelineRunState"),
    "PipelineRunAlreadyActiveError": ("library.core.errors", "PipelineRunAlreadyActiveError"),
    "PipelineStagePolicyContext": ("library.core.pipeline.PipelineExecutionPolicy", "PipelineStagePolicyContext"),
    "PluginConstructionError": ("library.core.errors", "PluginConstructionError"),
    "PluginResolutionError": ("library.core.errors", "PluginResolutionError"),
    "StageErrorContext": ("library.core.errors", "StageErrorContext"),
    "StageExecutionError": ("library.core.errors", "StageExecutionError"),
    "StreamAbortedError": ("library.core.errors", "StreamAbortedError"),
    "StreamRuntimeError": ("library.core.errors", "StreamRuntimeError"),
    "StreamRuntimeConfig": ("library.core.pipeline.StreamRuntimeConfig", "StreamRuntimeConfig"),
    "ThreadedPipelineRunner": ("library.core.pipeline.ThreadedPipelineRunner", "ThreadedPipelineRunner"),
    "VisualizerBinding": ("library.core.pipeline.VisualizerBinding", "VisualizerBinding"),
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


def __dir__() -> list[str]:
    return sorted(__all__)


_EAGER_EXPORTS = (
    "AdaptiveSamplingFrameLatencyPolicy",
    "BlockingFrameLatencyPolicy",
    "ConfigPipelineBuilder",
    "ConfigSchemaError",
    "DefaultPipelineExecutionPolicy",
    "DropNewestFrameLatencyPolicy",
    "DropOldestFrameLatencyPolicy",
    "ExecutionPlanStage",
    "FluentPipelineBuilder",
    "FrameLatencyPolicy",
    "LatencyPolicyConfig",
    "Pipeline",
    "PipelineConfigurationError",
    "PipelineContext",
    "PipelineContextError",
    "PipelineError",
    "PipelineExecutionDecision",
    "PipelineExecutionError",
    "PipelineExecutionMode",
    "PipelineExecutionPlan",
    "PipelineRunAlreadyActiveError",
    "PipelineRunSnapshot",
    "PipelineRunState",
    "PipelineStagePolicyContext",
    "PluginConstructionError",
    "PluginResolutionError",
    "StageErrorContext",
    "StageExecutionError",
    "StreamAbortedError",
    "StreamRuntimeConfig",
    "StreamRuntimeError",
    "ThreadedPipelineRunner",
    "VisualizerBinding",
)

for _name in _EAGER_EXPORTS:
    __getattr__(_name)
del _name
del _EAGER_EXPORTS
