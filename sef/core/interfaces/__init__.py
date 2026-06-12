"""Stable component and buffer interfaces for SEF plugins.

This package is the main authoring surface for plugin developers. It separates
batch contracts from streaming contracts, keeps component APIs independent from
UI frameworks, and exposes capability metadata used by the pipeline planner.

Design contract
---------------
Implementations should declare conservative `StageCapabilities`, preserve input
ordering unless explicitly documented otherwise, and keep runtime side effects
behind the appropriate output contracts rather than leaking framework-specific
objects into core values.
"""

from __future__ import annotations

from importlib import import_module

from sef.core._lazy_exports import install_lazy_exports

_EXPORTS = {
    "IAnalyzer": ("sef.core.interfaces.IAnalyzer", "IAnalyzer"),
    "IAbortableBuffer": ("sef.core.interfaces.BufferContracts", "IAbortableBuffer"),
    "IBranchingRule": ("sef.core.interfaces.pipeline.IBranchingRule", "IBranchingRule"),
    "IBuffer": ("sef.core.interfaces.BufferContracts", "IBuffer"),
    "IBufferSubscription": ("sef.core.interfaces.BufferContracts", "IBufferSubscription"),
    "IData": ("sef.core.interfaces.IData", "IData"),
    "IEventEmitter": ("sef.core.interfaces.IEventEmitter", "IEventEmitter"),
    "IEventBus": ("sef.core.interfaces.pipeline.IEventBus", "IEventBus"),
    "IFrameBuffer": ("sef.core.interfaces.BufferContracts", "IFrameBuffer"),
    "ILiveAnalyzer": ("sef.core.interfaces.ILiveAnalyzer", "ILiveAnalyzer"),
    "ISingleFrameProcessor": ("sef.core.interfaces.ISingleFrameProcessor", "ISingleFrameProcessor"),
    "IFrameExtractor": ("sef.core.interfaces.IFrameExtractor", "IFrameExtractor"),
    "IFrameBufferProcessor": ("sef.core.interfaces.IFrameBufferProcessor", "IFrameBufferProcessor"),
    "IFrameExporter": ("sef.core.interfaces.IFrameExporter", "IFrameExporter"),
    "FrameExportContext": ("sef.core.interfaces.IFrameExporter", "FrameExportContext"),
    "FrameExportResult": ("sef.core.interfaces.IFrameExporter", "FrameExportResult"),
    "StageCapabilities": ("sef.core.interfaces.StageCapabilities", "StageCapabilities"),
    "IStreamingFrameExtractor": ("sef.core.interfaces.StreamingContracts", "IStreamingFrameExtractor"),
    "IStreamingFrameBufferProcessor": ("sef.core.interfaces.StreamingContracts", "IStreamingFrameBufferProcessor"),
    "IStreamingFrameExporter": ("sef.core.interfaces.StreamingContracts", "IStreamingFrameExporter"),
    "IStreamingSignalExtractor": ("sef.core.interfaces.StreamingContracts", "IStreamingSignalExtractor"),
    "IStreamingSignalCleaner": ("sef.core.interfaces.StreamingContracts", "IStreamingSignalCleaner"),
    "IStreamingAnalyzer": ("sef.core.interfaces.StreamingContracts", "IStreamingAnalyzer"),
    "IStreamingVisualizer": ("sef.core.interfaces.StreamingContracts", "IStreamingVisualizer"),
    "IPipelineMonitor": (
        "sef.core.interfaces.pipeline.IPipelineMonitor",
        "IPipelineMonitor",
    ),
    "IPipelineOutputStore": (
        "sef.core.interfaces.pipeline.IPipelineOutputStore",
        "IPipelineOutputStore",
    ),
    "IPipelineFactory": (
        "sef.core.interfaces.pipeline.IPipelineFactory",
        "IPipelineFactory",
    ),
    "IPipelineRunner": (
        "sef.core.interfaces.pipeline.IPipelineRunner",
        "IPipelineRunner",
    ),
    "IPipelineValidator": (
        "sef.core.interfaces.pipeline.IPipelineValidator",
        "IPipelineValidator",
    ),
    "IRetryPolicy": ("sef.core.interfaces.pipeline.IRetryPolicy", "IRetryPolicy"),
    "ISignal": ("sef.core.interfaces.ISignal", "ISignal"),
    "ISignalCleaner": ("sef.core.interfaces.ISignalCleaner", "ISignalCleaner"),
    "ISignalExtractor": ("sef.core.interfaces.ISignalExtractor", "ISignalExtractor"),
    "ISignalSample": ("sef.core.interfaces.ISignalSample", "ISignalSample"),
    "ISubscribableBuffer": ("sef.core.interfaces.BufferContracts", "ISubscribableBuffer"),
    "IVisualizer": ("sef.core.interfaces.IVisualizer", "IVisualizer"),
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


