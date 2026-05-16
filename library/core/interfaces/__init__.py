from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "IAnalyzer": ("library.core.interfaces.IAnalyzer", "IAnalyzer"),
    "IBranchingRule": ("library.core.interfaces.pipeline.IBranchingRule", "IBranchingRule"),
    "IData": ("library.core.interfaces.IData", "IData"),
    "IEventEmitter": ("library.core.interfaces.IEventEmitter", "IEventEmitter"),
    "IEventBus": ("library.core.interfaces.pipeline.IEventBus", "IEventBus"),
    "ISingleFrameProcessor": ("library.core.interfaces.ISingleFrameProcessor", "ISingleFrameProcessor"),
    "IFrameExtractor": ("library.core.interfaces.IFrameExtractor", "IFrameExtractor"),
    "IFrameBufferProcessor": ("library.core.interfaces.IFrameBufferProcessor", "IFrameBufferProcessor"),
    "IFrameExporter": ("library.core.interfaces.IFrameExporter", "IFrameExporter"),
    "FrameExportContext": ("library.core.interfaces.IFrameExporter", "FrameExportContext"),
    "FrameExportResult": ("library.core.interfaces.IFrameExporter", "FrameExportResult"),
    "StageCapabilities": ("library.core.interfaces.StageCapabilities", "StageCapabilities"),
    "IStreamingFrameExtractor": ("library.core.interfaces.StreamingContracts", "IStreamingFrameExtractor"),
    "IStreamingFrameBufferProcessor": ("library.core.interfaces.StreamingContracts", "IStreamingFrameBufferProcessor"),
    "IStreamingFrameExporter": ("library.core.interfaces.StreamingContracts", "IStreamingFrameExporter"),
    "IStreamingSignalExtractor": ("library.core.interfaces.StreamingContracts", "IStreamingSignalExtractor"),
    "IStreamingSignalCleaner": ("library.core.interfaces.StreamingContracts", "IStreamingSignalCleaner"),
    "IStreamingAnalyzer": ("library.core.interfaces.StreamingContracts", "IStreamingAnalyzer"),
    "IStreamingVisualizer": ("library.core.interfaces.StreamingContracts", "IStreamingVisualizer"),
    "IPipelineMonitor": (
        "library.core.interfaces.pipeline.IPipelineMonitor",
        "IPipelineMonitor",
    ),
    "IPipelineFactory": (
        "library.core.interfaces.pipeline.IPipelineFactory",
        "IPipelineFactory",
    ),
    "IPipelineRunner": (
        "library.core.interfaces.pipeline.IPipelineRunner",
        "IPipelineRunner",
    ),
    "IPipelineValidator": (
        "library.core.interfaces.pipeline.IPipelineValidator",
        "IPipelineValidator",
    ),
    "ISignal": ("library.core.interfaces.ISignal", "ISignal"),
    "ISignalCleaner": ("library.core.interfaces.ISignalCleaner", "ISignalCleaner"),
    "ISignalExtractor": ("library.core.interfaces.ISignalExtractor", "ISignalExtractor"),
    "IVisualizer": ("library.core.interfaces.IVisualizer", "IVisualizer"),
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
