"""Public pipeline construction, planning, and execution contracts.

The package exposes the stable runtime API used by applications, services, and
UIs: `Pipeline`, `PipelineContext`, builders, execution policies, latency
policies, config-version helpers, runner snapshots, and asynchronous execution
facades.

Boundary
--------
Public users should depend on these exports instead of importing executor
collaborators directly. Executors and segment materializers are implementation
details; custom behavior should normally be introduced through component
interfaces, `PipelineExecutionPolicy`, `PluginRegistry`, or orchestration ports.
"""

from __future__ import annotations

from importlib import import_module

from sef.core._lazy_exports import install_lazy_exports

_EXPORTS = {
    "AdaptiveSamplingFrameLatencyPolicy": ("sef.core.pipeline.LatencyPolicy", "AdaptiveSamplingFrameLatencyPolicy"),
    "BlockingFrameLatencyPolicy": ("sef.core.pipeline.LatencyPolicy", "BlockingFrameLatencyPolicy"),
    "ConfigPipelineBuilder": ("sef.core.pipeline.ConfigPipelineBuilder", "ConfigPipelineBuilder"),
    "ConfigSchemaError": ("sef.core.errors", "ConfigSchemaError"),
    "ConfigVersionError": ("sef.core.errors", "ConfigVersionError"),
    "CURRENT_PIPELINE_CONFIG_VERSION": (
        "sef.core.pipeline.PipelineConfigVersioning",
        "CURRENT_PIPELINE_CONFIG_VERSION",
    ),
    "DefaultPipelineExecutionPolicy": ("sef.core.pipeline.PipelineExecutionPolicy", "DefaultPipelineExecutionPolicy"),
    "DropNewestFrameLatencyPolicy": ("sef.core.pipeline.LatencyPolicy", "DropNewestFrameLatencyPolicy"),
    "DropOldestFrameLatencyPolicy": ("sef.core.pipeline.LatencyPolicy", "DropOldestFrameLatencyPolicy"),
    "ExecutionPlanStage": ("sef.core.pipeline.PipelineExecutionPlan", "ExecutionPlanStage"),
    "FluentPipelineBuilder": ("sef.core.pipeline.FluentPipelineBuilder", "FluentPipelineBuilder"),
    "FrameLatencyPolicy": ("sef.core.pipeline.LatencyPolicy", "FrameLatencyPolicy"),
    "LatencyPolicyConfig": ("sef.core.pipeline.LatencyPolicy", "LatencyPolicyConfig"),
    "PIPELINE_CONFIG_VERSION_KEY": (
        "sef.core.pipeline.PipelineConfigVersioning",
        "PIPELINE_CONFIG_VERSION_KEY",
    ),
    "Pipeline": ("sef.core.pipeline.Pipeline", "Pipeline"),
    "PipelineConfigMigration": ("sef.core.pipeline.PipelineConfigVersioning", "PipelineConfigMigration"),
    "PipelineConfigVersionManager": (
        "sef.core.pipeline.PipelineConfigVersioning",
        "PipelineConfigVersionManager",
    ),
    "PipelineConfigurationError": ("sef.core.errors", "PipelineConfigurationError"),
    "PipelineContext": ("sef.core.pipeline.PipelineContext", "PipelineContext"),
    "PipelineContextError": ("sef.core.errors", "PipelineContextError"),
    "PipelineError": ("sef.core.errors", "PipelineError"),
    "PipelineExecutionDecision": ("sef.core.pipeline.PipelineExecutionPolicy", "PipelineExecutionDecision"),
    "PipelineExecutionError": ("sef.core.errors", "PipelineExecutionError"),
    "PipelineExecutionMode": ("sef.core.pipeline.PipelineExecutionPolicy", "PipelineExecutionMode"),
    "PipelineExecutionPlan": ("sef.core.pipeline.PipelineExecutionPlan", "PipelineExecutionPlan"),
    "PipelineOrchestrator": ("sef.core.pipeline.PipelineOrchestrator", "PipelineOrchestrator"),
    "PipelineRunSnapshot": ("sef.core.pipeline.PipelineRunSnapshot", "PipelineRunSnapshot"),
    "PipelineRunState": ("sef.core.pipeline.PipelineRunSnapshot", "PipelineRunState"),
    "PipelineRunAlreadyActiveError": ("sef.core.errors", "PipelineRunAlreadyActiveError"),
    "PipelineStagePolicyContext": ("sef.core.pipeline.PipelineExecutionPolicy", "PipelineStagePolicyContext"),
    "PluginConstructionError": ("sef.core.errors", "PluginConstructionError"),
    "PluginResolutionError": ("sef.core.errors", "PluginResolutionError"),
    "StageErrorContext": ("sef.core.errors", "StageErrorContext"),
    "StageExecutionError": ("sef.core.errors", "StageExecutionError"),
    "StreamAbortedError": ("sef.core.errors", "StreamAbortedError"),
    "StreamRuntimeError": ("sef.core.errors", "StreamRuntimeError"),
    "StreamRuntimeConfig": ("sef.core.pipeline.StreamRuntimeConfig", "StreamRuntimeConfig"),
    "ThreadedPipelineRunner": ("sef.core.pipeline.ThreadedPipelineRunner", "ThreadedPipelineRunner"),
    "VersionedPipelineConfig": ("sef.core.pipeline.PipelineConfigVersioning", "VersionedPipelineConfig"),
    "VisualizerBinding": ("sef.core.pipeline.VisualizerBinding", "VisualizerBinding"),
    "normalize_pipeline_config": (
        "sef.core.pipeline.PipelineConfigVersioning",
        "normalize_pipeline_config",
    ),
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


install_lazy_exports(__name__)
