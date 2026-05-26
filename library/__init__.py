from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "FluentPipelineBuilder": (
        "library.core.pipeline.FluentPipelineBuilder",
        "FluentPipelineBuilder",
    ),
    "ConfigPipelineBuilder": (
        "library.core.pipeline.ConfigPipelineBuilder",
        "ConfigPipelineBuilder",
    ),
    "PipelineOrchestrator": (
        "library.core.pipeline.PipelineOrchestrator",
        "PipelineOrchestrator",
    ),
    "BranchingCoordinator": (
        "library.core.pipeline.BranchingCoordinator",
        "BranchingCoordinator",
    ),
    "ThreadedPipelineRunner": (
        "library.core.pipeline.ThreadedPipelineRunner",
        "ThreadedPipelineRunner",
    ),
    "InMemoryPipelineMonitor": (
        "library.core.pipeline.InMemoryPipelineMonitor",
        "InMemoryPipelineMonitor",
    ),
    "DefaultPipelineFactory": (
        "library.core.pipeline.DefaultPipelineFactory",
        "DefaultPipelineFactory",
    ),
    "IEventBus": ("library.core.interfaces.pipeline.IEventBus", "IEventBus"),
    "IPipelineFactory": (
        "library.core.interfaces.pipeline.IPipelineFactory",
        "IPipelineFactory",
    ),
    "IPipelineRunner": (
        "library.core.interfaces.pipeline.IPipelineRunner",
        "IPipelineRunner",
    ),
    "IPipelineMonitor": (
        "library.core.interfaces.pipeline.IPipelineMonitor",
        "IPipelineMonitor",
    ),
    "Event": ("library.core.events.Event", "Event"),
    "EventBus": ("library.core.events.EventBus", "EventBus"),
    "PipelineEvent": ("library.core.events.PipelineEvent", "PipelineEvent"),
    "PipelineLifecycleEvent": (
        "library.core.events.PipelineLifecycleEvent",
        "PipelineLifecycleEvent",
    ),
    "PipelineRunSnapshot": (
        "library.core.pipeline.PipelineRunSnapshot",
        "PipelineRunSnapshot",
    ),
    "PipelineRunState": (
        "library.core.pipeline.PipelineRunSnapshot",
        "PipelineRunState",
    ),
    "Pipeline": ("library.core", "Pipeline"),
    "PipelineContext": ("library.core", "PipelineContext"),
    "PipelineError": ("library.core", "PipelineError"),
    "PipelineExecutionError": ("library.core", "PipelineExecutionError"),
    "PipelineContextError": ("library.core", "PipelineContextError"),
    "ConfigSchemaError": ("library.core", "ConfigSchemaError"),
    "ConfigVersionError": ("library.core", "ConfigVersionError"),
    "PluginResolutionError": ("library.core", "PluginResolutionError"),
    "PluginConstructionError": ("library.core", "PluginConstructionError"),
    "PluginRegistryError": ("library.core", "PluginRegistryError"),
    "InvalidPluginRegistrationError": ("library.core", "InvalidPluginRegistrationError"),
    "DuplicatePluginRegistrationError": ("library.core", "DuplicatePluginRegistrationError"),
    "SEFError": ("library.core", "SEFError"),
    "StageErrorContext": ("library.core", "StageErrorContext"),
    "StageExecutionError": ("library.core", "StageExecutionError"),
    "StreamAbortedError": ("library.core", "StreamAbortedError"),
    "StreamRuntimeError": ("library.core", "StreamRuntimeError"),
    "VisualizerBinding": (
        "library.core.pipeline.VisualizerBinding",
        "VisualizerBinding",
    ),
    "PipelineConfigurationError": (
        "library.core.pipeline.PipelineErrors",
        "PipelineConfigurationError",
    ),
    "InvalidPipelineTriggerEventError": (
        "library.core.pipeline.PipelineErrors",
        "InvalidPipelineTriggerEventError",
    ),
    "PipelineRunAlreadyActiveError": (
        "library.core.pipeline.PipelineErrors",
        "PipelineRunAlreadyActiveError",
    ),
    "StreamRuntimeConfig": ("library.core", "StreamRuntimeConfig"),
    "LatencyPolicyConfig": ("library.core", "LatencyPolicyConfig"),
    "CURRENT_PIPELINE_CONFIG_VERSION": ("library.core", "CURRENT_PIPELINE_CONFIG_VERSION"),
    "PIPELINE_CONFIG_VERSION_KEY": ("library.core", "PIPELINE_CONFIG_VERSION_KEY"),
    "create_pipeline_lifecycle_event": (
        "library.core.events.PipelineLifecycleEvent",
        "create_pipeline_lifecycle_event",
    ),
    "IEventEmitter": ("library.core.interfaces.IEventEmitter", "IEventEmitter"),
    "IBranchingRule": (
        "library.core.interfaces.pipeline.IBranchingRule",
        "IBranchingRule",
    ),
    "PluginCategory": ("library.core.plugins", "PluginCategory"),
    "PluginRegistry": ("library.core.plugins.PluginRegistry", "PluginRegistry"),
    "PluginDefinition": ("library.core.plugins", "PluginDefinition"),
    "StageCapabilities": ("library.core", "StageCapabilities"),
    "VisualArtifact": ("library.core", "VisualArtifact"),
    "TextArtifact": ("library.core", "TextArtifact"),
    "PipelineOutputs": ("library.core", "PipelineOutputs"),
    "VisualizationContext": ("library.core", "VisualizationContext"),
    "create_builtin_registry": (
        "library.core.plugins.PluginRegistry",
        "create_builtin_registry",
    ),
    "VerticalPositionAnalyzer": (
        "library.analyzers.VerticalPositionAnalyzer",
        "VerticalPositionAnalyzer",
    ),
    "OpenCVBufferedFrameExtractor": (
        "library.frame_extractors.OpenCVBufferedFrameExtractor",
        "OpenCVBufferedFrameExtractor",
    ),
    "OpenCVBufferedSignalExtractor": (
        "library.signal_extractors.OpenCVBufferedSignalExtractor",
        "OpenCVBufferedSignalExtractor",
    ),
    "MovingAverageCleaner": (
        "library.signal_cleaners.MovingAverageCleaner",
        "MovingAverageCleaner",
    ),
    "MatplotlibFunctionVisualizer": (
        "library.visualizers.MatplotlibFunctionVisualizer",
        "MatplotlibFunctionVisualizer",
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
