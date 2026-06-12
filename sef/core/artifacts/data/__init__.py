"""Public analysis data artifacts produced by analyzers and visualizers."""

from __future__ import annotations

from importlib import import_module

from sef.core._lazy_exports import install_lazy_exports

_EXPORTS = {
    "ArucoMarkerDisplacementData": (
        "sef.core.artifacts.data.ArucoDisplacementData",
        "ArucoMarkerDisplacementData",
    ),
    "ArucoMarkerDisplacementFrameData": (
        "sef.core.artifacts.data.ArucoDisplacementData",
        "ArucoMarkerDisplacementFrameData",
    ),
    "ArucoMarkerDisplacementObservation": (
        "sef.core.artifacts.data.ArucoDisplacementData",
        "ArucoMarkerDisplacementObservation",
    ),
    "ArucoMarkerDisplacementSeries": (
        "sef.core.artifacts.data.ArucoDisplacementData",
        "ArucoMarkerDisplacementSeries",
    ),
    "ArucoMarkerRelativeMotionData": (
        "sef.core.artifacts.data.ArucoRelativeMotionData",
        "ArucoMarkerRelativeMotionData",
    ),
    "ArucoMarkerRelativeMotionSeries": (
        "sef.core.artifacts.data.ArucoRelativeMotionData",
        "ArucoMarkerRelativeMotionSeries",
    ),
    "CategoryData": ("sef.core.artifacts.data.CategoryData", "CategoryData"),
    "COCOPoseFrameData": ("sef.core.artifacts.data.COCOPoseFrameData", "COCOPoseFrameData"),
    "COCOPoseSequenceData": (
        "sef.core.artifacts.data.COCOPoseFrameData",
        "COCOPoseSequenceData",
    ),
    "COCOPoseTennisFrameData": (
        "sef.core.artifacts.data.COCOPoseTennisFrameData",
        "COCOPoseTennisFrameData",
    ),
    "COCOPoseTennisSequenceData": (
        "sef.core.artifacts.data.COCOPoseTennisFrameData",
        "COCOPoseTennisSequenceData",
    ),
    "NoData": ("sef.core.artifacts.data.NoData", "NoData"),
    "TrackingPlaybackData": (
        "sef.core.artifacts.data.TrackingPlaybackData",
        "TrackingPlaybackData",
    ),
    "TrackingPlaybackFrame": (
        "sef.core.artifacts.data.TrackingPlaybackData",
        "TrackingPlaybackFrame",
    ),
    "TrackingPlaybackTrack": (
        "sef.core.artifacts.data.TrackingPlaybackData",
        "TrackingPlaybackTrack",
    ),
    "TrajectoryData": ("sef.core.artifacts.data.TrajectoryData", "TrajectoryData"),
    "TwoDimGraphData": ("sef.core.artifacts.data.TwoDimGraphData", "TwoDimGraphData"),
    "TwoDimPointData": ("sef.core.artifacts.data.TwoDimPointData", "TwoDimPointData"),
    "VectorFieldGraphData": (
        "sef.core.artifacts.data.VectorFieldGraphData",
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


install_lazy_exports(__name__)
