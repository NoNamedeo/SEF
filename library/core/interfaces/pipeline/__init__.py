from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "IBranchingRule": (
        "library.core.interfaces.pipeline.IBranchingRule",
        "IBranchingRule",
    ),
    "IEventBus": ("library.core.interfaces.pipeline.IEventBus", "IEventBus"),
    "IPipelineFactory": (
        "library.core.interfaces.pipeline.IPipelineFactory",
        "IPipelineFactory",
    ),
    "IPipelineMonitor": (
        "library.core.interfaces.pipeline.IPipelineMonitor",
        "IPipelineMonitor",
    ),
    "IPipelineOutputStore": (
        "library.core.interfaces.pipeline.IPipelineOutputStore",
        "IPipelineOutputStore",
    ),
    "IPipelineRunner": (
        "library.core.interfaces.pipeline.IPipelineRunner",
        "IPipelineRunner",
    ),
    "IPipelineValidator": (
        "library.core.interfaces.pipeline.IPipelineValidator",
        "IPipelineValidator",
    ),
    "IRetryPolicy": ("library.core.interfaces.pipeline.IRetryPolicy", "IRetryPolicy"),
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
