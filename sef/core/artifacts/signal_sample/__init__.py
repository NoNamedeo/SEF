"""Public signal sample values produced by signal extractors."""

from __future__ import annotations

from importlib import import_module

from sef.core._lazy_exports import install_lazy_exports

_EXPORTS = {
    "ArucoMarkerObservation": (
        "sef.core.artifacts.signal_sample.ArucoMarkerSignalSample",
        "ArucoMarkerObservation",
    ),
    "ArucoMarkerSignalSample": (
        "sef.core.artifacts.signal_sample.ArucoMarkerSignalSample",
        "ArucoMarkerSignalSample",
    ),
    "BoxSignalSample": ("sef.core.artifacts.signal_sample.BoxSignalSample", "BoxSignalSample"),
    "COCOSkeletonSignalSample": (
        "sef.core.artifacts.signal_sample.COCOSkeletonSignalSample",
        "COCOSkeletonSignalSample",
    ),
    "DenseOpticalFlowSignalSample": (
        "sef.core.artifacts.signal_sample.DenseOpticalFlowSignalSample",
        "DenseOpticalFlowSignalSample",
    ),
    "MultiManualSignalSample": (
        "sef.core.artifacts.signal_sample.MultiManualSignalSample",
        "MultiManualSignalSample",
    ),
    "MultiObjectSignalSample": (
        "sef.core.artifacts.signal_sample.MultiObjectSignalSample",
        "MultiObjectSignalSample",
    ),
    "MultiObjectTrack": (
        "sef.core.artifacts.signal_sample.MultiObjectSignalSample",
        "MultiObjectTrack",
    ),
    "SparseOpticalFlowSignalSample": (
        "sef.core.artifacts.signal_sample.SparseOpticalFlowSignalSample",
        "SparseOpticalFlowSignalSample",
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


install_lazy_exports(__name__)
