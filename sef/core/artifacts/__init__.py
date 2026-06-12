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

from sef.core._lazy_exports import install_lazy_exports

_EXPORTS = {
    "ArucoMarkerDisplacementData": (
        "sef.core.artifacts.data",
        "ArucoMarkerDisplacementData",
    ),
    "ArucoMarkerDisplacementFrameData": (
        "sef.core.artifacts.data",
        "ArucoMarkerDisplacementFrameData",
    ),
    "ArucoMarkerDisplacementObservation": (
        "sef.core.artifacts.data",
        "ArucoMarkerDisplacementObservation",
    ),
    "ArucoMarkerDisplacementSeries": (
        "sef.core.artifacts.data",
        "ArucoMarkerDisplacementSeries",
    ),
    "ArucoMarkerObservation": (
        "sef.core.artifacts.signal_sample",
        "ArucoMarkerObservation",
    ),
    "ArucoMarkerRelativeMotionData": (
        "sef.core.artifacts.data",
        "ArucoMarkerRelativeMotionData",
    ),
    "ArucoMarkerRelativeMotionSeries": (
        "sef.core.artifacts.data",
        "ArucoMarkerRelativeMotionSeries",
    ),
    "ArucoMarkerSignalSample": (
        "sef.core.artifacts.signal_sample",
        "ArucoMarkerSignalSample",
    ),
    "BoxSignalSample": ("sef.core.artifacts.signal_sample", "BoxSignalSample"),
    "CategoryData": ("sef.core.artifacts.data", "CategoryData"),
    "COCOPoseFrameData": ("sef.core.artifacts.data", "COCOPoseFrameData"),
    "COCOPoseSequenceData": (
        "sef.core.artifacts.data",
        "COCOPoseSequenceData",
    ),
    "COCOPoseTennisFrameData": (
        "sef.core.artifacts.data",
        "COCOPoseTennisFrameData",
    ),
    "COCOPoseTennisSequenceData": (
        "sef.core.artifacts.data",
        "COCOPoseTennisSequenceData",
    ),
    "COCOSkeletonSignalSample": (
        "sef.core.artifacts.signal_sample",
        "COCOSkeletonSignalSample",
    ),
    "DataBuffer": ("sef.core.artifacts.buffer", "DataBuffer"),
    "DataSubscription": ("sef.core.artifacts.buffer", "DataSubscription"),
    "DenseOpticalFlowSignalSample": (
        "sef.core.artifacts.signal_sample",
        "DenseOpticalFlowSignalSample",
    ),
    "Frame": ("sef.core.artifacts.Frame", "Frame"),
    "FrameBuffer": ("sef.core.artifacts.buffer", "FrameBuffer"),
    "FrameComparisonPanel": (
        "sef.core.artifacts.intermediate_frame",
        "FrameComparisonPanel",
    ),
    "FrameMaskArtifact": ("sef.core.artifacts.mask", "FrameMaskArtifact"),
    "IntermediateFrameArtifact": (
        "sef.core.artifacts.mask",
        "IntermediateFrameArtifact",
    ),
    "IntermediateFrameArtifactCollection": (
        "sef.core.artifacts.intermediate_frame",
        "IntermediateFrameArtifactCollection",
    ),
    "IntermediateFrameOverlay": (
        "sef.core.artifacts.mask",
        "IntermediateFrameOverlay",
    ),
    "MaskArtifact": ("sef.core.artifacts.mask", "MaskArtifact"),
    "MotionMaskArtifact": ("sef.core.artifacts.mask", "MotionMaskArtifact"),
    "MultiManualSignalSample": (
        "sef.core.artifacts.signal_sample",
        "MultiManualSignalSample",
    ),
    "MultiObjectSignalSample": (
        "sef.core.artifacts.signal_sample",
        "MultiObjectSignalSample",
    ),
    "MultiObjectTrack": ("sef.core.artifacts.signal_sample", "MultiObjectTrack"),
    "NoData": ("sef.core.artifacts.data", "NoData"),
    "ProtectedRegionArtifact": (
        "sef.core.artifacts.mask",
        "ProtectedRegionArtifact",
    ),
    "Signal": ("sef.core.artifacts.Signal", "Signal"),
    "SignalBuffer": ("sef.core.artifacts.buffer", "SignalBuffer"),
    "SignalSubscription": ("sef.core.artifacts.buffer", "SignalSubscription"),
    "SparseOpticalFlowSignalSample": (
        "sef.core.artifacts.signal_sample",
        "SparseOpticalFlowSignalSample",
    ),
    "TargetMaskArtifact": ("sef.core.artifacts.mask", "TargetMaskArtifact"),
    "TrackingPlaybackData": (
        "sef.core.artifacts.data",
        "TrackingPlaybackData",
    ),
    "TrackingPlaybackFrame": (
        "sef.core.artifacts.data",
        "TrackingPlaybackFrame",
    ),
    "TrackingPlaybackTrack": (
        "sef.core.artifacts.data",
        "TrackingPlaybackTrack",
    ),
    "TrajectoryData": ("sef.core.artifacts.data", "TrajectoryData"),
    "TwoDimGraphData": ("sef.core.artifacts.data", "TwoDimGraphData"),
    "TwoDimPointData": ("sef.core.artifacts.data", "TwoDimPointData"),
    "VectorFieldGraphData": (
        "sef.core.artifacts.data",
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


