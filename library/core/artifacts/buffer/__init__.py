"""Public frame, signal, and data buffers used across the pipeline."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DataBuffer": ("library.core.artifacts.buffer.DataBuffer", "DataBuffer"),
    "DataSubscription": ("library.core.artifacts.buffer.DataBuffer", "DataSubscription"),
    "FrameBuffer": ("library.core.artifacts.buffer.FrameBuffer", "FrameBuffer"),
    "SignalBuffer": ("library.core.artifacts.buffer.SignalBuffer", "SignalBuffer"),
    "SignalSubscription": ("library.core.artifacts.buffer.SignalBuffer", "SignalSubscription"),
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


for _name in __all__:
    __getattr__(_name)
del _name
