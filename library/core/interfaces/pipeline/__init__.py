from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "IEventBus": ("library.core.interfaces.pipeline.IEventBus", "IEventBus"),
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
