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
    "create_pipeline_lifecycle_event": (
        "library.core.events.PipelineLifecycleEvent",
        "create_pipeline_lifecycle_event",
    ),
    "IEventEmitter": ("library.core.interfaces.IEventEmitter", "IEventEmitter"),
    "IBranchingRule": (
        "library.core.interfaces.pipeline.IBranchingRule",
        "IBranchingRule",
    ),
    "PluginRegistry": ("library.core.plugins.PluginRegistry", "PluginRegistry"),
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
