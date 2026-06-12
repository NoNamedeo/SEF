"""Public frame, signal, and data buffers used across the pipeline."""

from __future__ import annotations

from importlib import import_module

from sef.core._lazy_exports import install_lazy_exports

_EXPORTS = {
    "DataBuffer": ("sef.core.artifacts.buffer.DataBuffer", "DataBuffer"),
    "DataSubscription": ("sef.core.artifacts.buffer.DataBuffer", "DataSubscription"),
    "FrameBuffer": ("sef.core.artifacts.buffer.FrameBuffer", "FrameBuffer"),
    "SignalBuffer": ("sef.core.artifacts.buffer.SignalBuffer", "SignalBuffer"),
    "SignalSubscription": ("sef.core.artifacts.buffer.SignalBuffer", "SignalSubscription"),
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
