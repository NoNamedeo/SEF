"""Public event contracts for pipeline triggers and lifecycle notifications.

The events package defines a small immutable event object, a thread-safe event
bus, and helper contracts for triggering pipeline execution or observing runner
lifecycle transitions. Event payloads should remain explicit and serializable
where possible so service, UI, and test adapters can inspect them safely.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Event": ("library.core.events.Event", "Event"),
    "EventBus": ("library.core.events.EventBus", "EventBus"),
    "EventHandler": ("library.core.events.EventBus", "EventHandler"),
    "PipelineEvent": ("library.core.events.PipelineEvent", "PipelineEvent"),
    "PipelineLifecycleEvent": (
        "library.core.events.PipelineLifecycleEvent",
        "PipelineLifecycleEvent",
    ),
    "create_pipeline_lifecycle_event": (
        "library.core.events.PipelineLifecycleEvent",
        "create_pipeline_lifecycle_event",
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
