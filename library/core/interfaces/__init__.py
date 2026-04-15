from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "IAnalyzer": ("library.core.interfaces.IAnalyzer", "IAnalyzer"),
    "IBranchingRule": ("library.core.interfaces.pipeline.IBranchingRule", "IBranchingRule"),
    "IData": ("library.core.interfaces.IData", "IData"),
    "IEventEmitter": ("library.core.interfaces.IEventEmitter", "IEventEmitter"),
    "IEventBus": ("library.core.interfaces.pipeline.IEventBus", "IEventBus"),
    "IFrameCleaner": ("library.core.interfaces.IFrameCleaner", "IFrameCleaner"),
    "IFrameExtractor": ("library.core.interfaces.IFrameExtractor", "IFrameExtractor"),
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
