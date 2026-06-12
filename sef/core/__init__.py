"""Expert API re-exporting the stable SEF core contracts."""

from __future__ import annotations

from importlib import import_module

_EXPORT_PACKAGES = (
    "artifacts",
    "errors",
    "events",
    "interfaces",
    "pipeline",
    "plugins",
    "visualization",
)

__all__ = [
    "BranchingCoordinator",
    "CURRENT_PIPELINE_CONFIG_VERSION",
    "ConfigPipelineBuilder",
    "ConfigSchemaError",
    "ConfigVersionError",
    "Event",
    "EventBus",
    "Frame",
    "FrameBuffer",
    "IBranchingRule",
    "IEventEmitter",
    "NoRetryPolicy",
    "Pipeline",
    "PipelineConfigurationError",
    "PipelineContext",
    "PipelineContextError",
    "PipelineEvent",
    "PipelineExecutionError",
    "PipelineLifecycleEvent",
    "PipelineOrchestrator",
    "PluginCategory",
    "PluginRegistry",
    "PluginResolutionError",
    "SEFError",
    "Signal",
    "StageErrorContext",
    "StageExecutionError",
    "TextArtifact",
    "ThreadedPipelineRunner",
    "VisualArtifact",
]


def __getattr__(name: str):
    for package_name in _EXPORT_PACKAGES:
        package = import_module(f"sef.core.{package_name}")
        if name in getattr(package, "__all__", ()):
            resolver = getattr(package, "__getattr__", None)
            value = resolver(name) if resolver is not None else getattr(package, name)
            globals()[name] = value
            return value
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
