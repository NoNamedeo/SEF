"""Public orchestration ports used by pipeline application services.

The interfaces in this package allow applications to replace monitoring,
output storage, runner execution, event buses, retry policy, validation, and
pipeline factories without depending on concrete infrastructure classes.

These contracts sit at the application boundary: implementations may use
threads, databases, web frameworks, or queues, but callers should interact only
through these ports.
"""

from __future__ import annotations

from importlib import import_module

from sef.core._lazy_exports import install_lazy_exports

_EXPORTS = {
    "IBranchingRule": (
        "sef.core.interfaces.pipeline.IBranchingRule",
        "IBranchingRule",
    ),
    "IEventBus": ("sef.core.interfaces.pipeline.IEventBus", "IEventBus"),
    "IPipelineFactory": (
        "sef.core.interfaces.pipeline.IPipelineFactory",
        "IPipelineFactory",
    ),
    "IPipelineMonitor": (
        "sef.core.interfaces.pipeline.IPipelineMonitor",
        "IPipelineMonitor",
    ),
    "IPipelineOutputStore": (
        "sef.core.interfaces.pipeline.IPipelineOutputStore",
        "IPipelineOutputStore",
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
