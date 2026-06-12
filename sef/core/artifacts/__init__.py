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
        "library.core.artifacts.data",
        "ArucoMarkerDisplacementData",
    ),
    "ArucoMarkerDisplacementFrameData": (
        "library.core.artifacts.data",
        "ArucoMarkerDisplacementFrameData",
    ),
    "ArucoMarkerDisplacementObservation": (
        "library.core.artifacts.data",
        "ArucoMarkerDisplacementObservation",
    ),
    "ArucoMarkerDisplacementSeries": (
        "library.core.artifacts.data",
        "ArucoMarkerDisplacementSeries",
    ),
    "ArucoMarkerObservation": (
        "library.core.artifacts.signal_sample",
        "ArucoMarkerObservation",
    ),
    "ArucoMarkerRelativeMotionData": (
        "library.core.artifacts.data",
        "ArucoMarkerRelativeMotionData",
    ),
    "ArucoMarkerRelativeMotionSeries": (
        "library.core.artifacts.data",
        "ArucoMarkerRelativeMotionSeries",
    ),
    "ArucoMarkerSignalSample": (
        "library.core.artifacts.signal_sample",
        "ArucoMarkerSignalSample",
    ),
    "BoxSignalSample": ("library.core.artifacts.signal_sample", "BoxSignalSample"),
    "CategoryData": ("library.core.artifacts.data", "CategoryData"),
    "COCOPoseFrameData": ("library.core.artifacts.data", "COCOPoseFrameData"),
    "COCOPoseSequenceData": (
        "library.core.artifacts.data",
        "COCOPoseSequenceData",
    ),
    "COCOPoseTennisFrameData": (
        "library.core.artifacts.data",
        "COCOPoseTennisFrameData",
    ),
    "COCOPoseTennisSequenceData": (
        "library.core.artifacts.data",
        "COCOPoseTennisSequenceData",
    ),
    "COCOSkeletonSignalSample": (
        "library.core.artifacts.signal_sample",
        "COCOSkeletonSignalSample",
    ),
    "DataBuffer": ("library.core.artifacts.buffer", "DataBuffer"),
    "DataSubscription": ("library.core.artifacts.buffer", "DataSubscription"),
    "DenseOpticalFlowSignalSample": (
        "library.core.artifacts.signal_sample",
        "DenseOpticalFlowSignalSample",
    ),
    "Frame": ("library.core.artifacts.Frame", "Frame"),
    "FrameBuffer": ("library.core.artifacts.buffer", "FrameBuffer"),
    "FrameComparisonPanel": (
        "library.core.artifacts.intermediate_frame",
        "FrameComparisonPanel",
    ),
    "FrameMaskArtifact": ("library.core.artifacts.mask", "FrameMaskArtifact"),
    "IntermediateFrameArtifact": (
        "library.core.artifacts.mask",
        "IntermediateFrameArtifact",
    ),
    "IntermediateFrameArtifactCollection": (
        "library.core.artifacts.intermediate_frame",
        "IntermediateFrameArtifactCollection",
    ),
    "IntermediateFrameOverlay": (
        "library.core.artifacts.mask",
        "IntermediateFrameOverlay",
    ),
    "MaskArtifact": ("library.core.artifacts.mask", "MaskArtifact"),
    "MotionMaskArtifact": ("library.core.artifacts.mask", "MotionMaskArtifact"),
    "MultiManualSignalSample": (
        "library.core.artifacts.signal_sample",
        "MultiManualSignalSample",
    ),
    "MultiObjectSignalSample": (
        "library.core.artifacts.signal_sample",
        "MultiObjectSignalSample",
    ),
    "MultiObjectTrack": ("library.core.artifacts.signal_sample", "MultiObjectTrack"),
    "NoData": ("library.core.artifacts.data", "NoData"),
    "ProtectedRegionArtifact": (
        "library.core.artifacts.mask",
        "ProtectedRegionArtifact",
    ),
    "Signal": ("library.core.artifacts.Signal", "Signal"),
    "SignalBuffer": ("library.core.artifacts.buffer", "SignalBuffer"),
    "SignalSubscription": ("library.core.artifacts.buffer", "SignalSubscription"),
    "SparseOpticalFlowSignalSample": (
        "library.core.artifacts.signal_sample",
        "SparseOpticalFlowSignalSample",
    ),
    "TargetMaskArtifact": ("library.core.artifacts.mask", "TargetMaskArtifact"),
    "TrackingPlaybackData": (
        "library.core.artifacts.data",
        "TrackingPlaybackData",
    ),
    "TrackingPlaybackFrame": (
        "library.core.artifacts.data",
        "TrackingPlaybackFrame",
    ),
    "TrackingPlaybackTrack": (
        "library.core.artifacts.data",
        "TrackingPlaybackTrack",
    ),
    "TrajectoryData": ("library.core.artifacts.data", "TrajectoryData"),
    "TwoDimGraphData": ("library.core.artifacts.data", "TwoDimGraphData"),
    "TwoDimPointData": ("library.core.artifacts.data", "TwoDimPointData"),
    "VectorFieldGraphData": (
        "library.core.artifacts.data",
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
