"""Public signal sample values produced by signal extractors."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ArucoMarkerObservation": (
        "library.core.artifacts.signal_sample.ArucoMarkerSignalSample",
        "ArucoMarkerObservation",
    ),
    "ArucoMarkerSignalSample": (
        "library.core.artifacts.signal_sample.ArucoMarkerSignalSample",
        "ArucoMarkerSignalSample",
    ),
    "BoxSignalSample": ("library.core.artifacts.signal_sample.BoxSignalSample", "BoxSignalSample"),
    "COCOSkeletonSignalSample": (
        "library.core.artifacts.signal_sample.COCOSkeletonSignalSample",
        "COCOSkeletonSignalSample",
    ),
    "DenseOpticalFlowSignalSample": (
        "library.core.artifacts.signal_sample.DenseOpticalFlowSignalSample",
        "DenseOpticalFlowSignalSample",
    ),
    "MultiManualSignalSample": (
        "library.core.artifacts.signal_sample.MultiManualSignalSample",
        "MultiManualSignalSample",
    ),
    "MultiObjectSignalSample": (
        "library.core.artifacts.signal_sample.MultiObjectSignalSample",
        "MultiObjectSignalSample",
    ),
    "MultiObjectTrack": (
        "library.core.artifacts.signal_sample.MultiObjectSignalSample",
        "MultiObjectTrack",
    ),
    "SparseOpticalFlowSignalSample": (
        "library.core.artifacts.signal_sample.SparseOpticalFlowSignalSample",
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


for _name in __all__:
    __getattr__(_name)
del _name
