"""Public data values exchanged by SEF pipeline stages.

Artifacts are intentionally framework-neutral. They model frames, signals,
sample values, analysis results, tracking playback, pose data, mask debug
outputs, and bounded stream buffers without importing UI or infrastructure
adapters.

Plugin authors should return these values, or project-specific subclasses of
the public interfaces, so downstream analyzers and visualizers can compose
without knowing which concrete extractor or processor produced the data.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ArucoMarkerDisplacementData": (
        "library.core.artifacts.ArucoDisplacementData",
        "ArucoMarkerDisplacementData",
    ),
    "ArucoMarkerDisplacementFrameData": (
        "library.core.artifacts.ArucoDisplacementData",
        "ArucoMarkerDisplacementFrameData",
    ),
    "ArucoMarkerDisplacementObservation": (
        "library.core.artifacts.ArucoDisplacementData",
        "ArucoMarkerDisplacementObservation",
    ),
    "ArucoMarkerDisplacementSeries": (
        "library.core.artifacts.ArucoDisplacementData",
        "ArucoMarkerDisplacementSeries",
    ),
    "ArucoMarkerObservation": (
        "library.core.artifacts.ArucoMarkerSignalSample",
        "ArucoMarkerObservation",
    ),
    "ArucoMarkerRelativeMotionData": (
        "library.core.artifacts.ArucoRelativeMotionData",
        "ArucoMarkerRelativeMotionData",
    ),
    "ArucoMarkerRelativeMotionSeries": (
        "library.core.artifacts.ArucoRelativeMotionData",
        "ArucoMarkerRelativeMotionSeries",
    ),
    "ArucoMarkerSignalSample": (
        "library.core.artifacts.ArucoMarkerSignalSample",
        "ArucoMarkerSignalSample",
    ),
    "BoxSignalSample": ("library.core.artifacts.BoxSignalSample", "BoxSignalSample"),
    "CategoryData": ("library.core.artifacts.CategoryData", "CategoryData"),
    "COCOPoseFrameData": ("library.core.artifacts.COCOPoseFrameData", "COCOPoseFrameData"),
    "COCOPoseSequenceData": (
        "library.core.artifacts.COCOPoseFrameData",
        "COCOPoseSequenceData",
    ),
    "COCOPoseTennisFrameData": (
        "library.core.artifacts.COCOPoseTennisFrameData",
        "COCOPoseTennisFrameData",
    ),
    "COCOPoseTennisSequenceData": (
        "library.core.artifacts.COCOPoseTennisFrameData",
        "COCOPoseTennisSequenceData",
    ),
    "COCOSkeletonSignalSample": (
        "library.core.artifacts.COCOSkeletonSignalSample",
        "COCOSkeletonSignalSample",
    ),
    "DataBuffer": ("library.core.artifacts.DataBuffer", "DataBuffer"),
    "DataSubscription": ("library.core.artifacts.DataBuffer", "DataSubscription"),
    "DenseOpticalFlowSignalSample": (
        "library.core.artifacts.DenseOpticalFlowSignalSample",
        "DenseOpticalFlowSignalSample",
    ),
    "Frame": ("library.core.artifacts.Frame", "Frame"),
    "FrameBuffer": ("library.core.artifacts.FrameBuffer", "FrameBuffer"),
    "FrameComparisonPanel": (
        "library.core.artifacts.IntermediateFrameComposition",
        "FrameComparisonPanel",
    ),
    "FrameMaskArtifact": ("library.core.artifacts.MaskArtifacts", "FrameMaskArtifact"),
    "IntermediateFrameArtifact": (
        "library.core.artifacts.MaskArtifacts",
        "IntermediateFrameArtifact",
    ),
    "IntermediateFrameArtifactCollection": (
        "library.core.artifacts.IntermediateFrameArtifacts",
        "IntermediateFrameArtifactCollection",
    ),
    "IntermediateFrameOverlay": (
        "library.core.artifacts.MaskArtifacts",
        "IntermediateFrameOverlay",
    ),
    "MaskArtifact": ("library.core.artifacts.MaskArtifacts", "MaskArtifact"),
    "MotionMaskArtifact": ("library.core.artifacts.MaskArtifacts", "MotionMaskArtifact"),
    "MultiManualSignalSample": (
        "library.core.artifacts.MultiManualSignalSample",
        "MultiManualSignalSample",
    ),
    "MultiObjectSignalSample": (
        "library.core.artifacts.MultiObjectSignalSample",
        "MultiObjectSignalSample",
    ),
    "MultiObjectTrack": ("library.core.artifacts.MultiObjectSignalSample", "MultiObjectTrack"),
    "NoData": ("library.core.artifacts.NoData", "NoData"),
    "ProtectedRegionArtifact": (
        "library.core.artifacts.MaskArtifacts",
        "ProtectedRegionArtifact",
    ),
    "Signal": ("library.core.artifacts.Signal", "Signal"),
    "SignalBuffer": ("library.core.artifacts.SignalBuffer", "SignalBuffer"),
    "SignalSubscription": ("library.core.artifacts.SignalBuffer", "SignalSubscription"),
    "SparseOpticalFlowSignalSample": (
        "library.core.artifacts.SparseOpticalFlowSignalSample",
        "SparseOpticalFlowSignalSample",
    ),
    "TargetMaskArtifact": ("library.core.artifacts.MaskArtifacts", "TargetMaskArtifact"),
    "TrackingPlaybackData": (
        "library.core.artifacts.TrackingPlaybackData",
        "TrackingPlaybackData",
    ),
    "TrackingPlaybackFrame": (
        "library.core.artifacts.TrackingPlaybackData",
        "TrackingPlaybackFrame",
    ),
    "TrackingPlaybackTrack": (
        "library.core.artifacts.TrackingPlaybackData",
        "TrackingPlaybackTrack",
    ),
    "TrajectoryData": ("library.core.artifacts.TrajectoryData", "TrajectoryData"),
    "TwoDimGraphData": ("library.core.artifacts.TwoDimGraphData", "TwoDimGraphData"),
    "TwoDimPointData": ("library.core.artifacts.TwoDimPointData", "TwoDimPointData"),
    "VectorFieldGraphData": (
        "library.core.artifacts.VectorFieldGraphData",
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


_EAGER_EXPORTS = tuple(name for name in __all__ if name != "FrameComparisonPanel")
for _name in _EAGER_EXPORTS:
    __getattr__(_name)
del _name
del _EAGER_EXPORTS
