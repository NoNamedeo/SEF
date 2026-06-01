"""Public analysis data artifacts produced by analyzers and visualizers."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ArucoMarkerDisplacementData": (
        "library.core.artifacts.data.ArucoDisplacementData",
        "ArucoMarkerDisplacementData",
    ),
    "ArucoMarkerDisplacementFrameData": (
        "library.core.artifacts.data.ArucoDisplacementData",
        "ArucoMarkerDisplacementFrameData",
    ),
    "ArucoMarkerDisplacementObservation": (
        "library.core.artifacts.data.ArucoDisplacementData",
        "ArucoMarkerDisplacementObservation",
    ),
    "ArucoMarkerDisplacementSeries": (
        "library.core.artifacts.data.ArucoDisplacementData",
        "ArucoMarkerDisplacementSeries",
    ),
    "ArucoMarkerRelativeMotionData": (
        "library.core.artifacts.data.ArucoRelativeMotionData",
        "ArucoMarkerRelativeMotionData",
    ),
    "ArucoMarkerRelativeMotionSeries": (
        "library.core.artifacts.data.ArucoRelativeMotionData",
        "ArucoMarkerRelativeMotionSeries",
    ),
    "CategoryData": ("library.core.artifacts.data.CategoryData", "CategoryData"),
    "COCOPoseFrameData": ("library.core.artifacts.data.COCOPoseFrameData", "COCOPoseFrameData"),
    "COCOPoseSequenceData": (
        "library.core.artifacts.data.COCOPoseFrameData",
        "COCOPoseSequenceData",
    ),
    "COCOPoseTennisFrameData": (
        "library.core.artifacts.data.COCOPoseTennisFrameData",
        "COCOPoseTennisFrameData",
    ),
    "COCOPoseTennisSequenceData": (
        "library.core.artifacts.data.COCOPoseTennisFrameData",
        "COCOPoseTennisSequenceData",
    ),
    "NoData": ("library.core.artifacts.data.NoData", "NoData"),
    "TrackingPlaybackData": (
        "library.core.artifacts.data.TrackingPlaybackData",
        "TrackingPlaybackData",
    ),
    "TrackingPlaybackFrame": (
        "library.core.artifacts.data.TrackingPlaybackData",
        "TrackingPlaybackFrame",
    ),
    "TrackingPlaybackTrack": (
        "library.core.artifacts.data.TrackingPlaybackData",
        "TrackingPlaybackTrack",
    ),
    "TrajectoryData": ("library.core.artifacts.data.TrajectoryData", "TrajectoryData"),
    "TwoDimGraphData": ("library.core.artifacts.data.TwoDimGraphData", "TwoDimGraphData"),
    "TwoDimPointData": ("library.core.artifacts.data.TwoDimPointData", "TwoDimPointData"),
    "VectorFieldGraphData": (
        "library.core.artifacts.data.VectorFieldGraphData",
        "VectorFieldGraphData",
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
