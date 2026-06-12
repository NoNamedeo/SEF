# API: Artifacts Package

Artifacts are typed values exchanged between stages and returned to adapters.
They are intentionally framework-neutral.

## Frames and Buffers

::: sef.core.artifacts.Frame.Frame

::: sef.core.artifacts.buffer.FrameBuffer.FrameBuffer

::: sef.core.artifacts.buffer.SignalBuffer.SignalBuffer

::: sef.core.artifacts.buffer.SignalBuffer.SignalSubscription

::: sef.core.artifacts.buffer.DataBuffer.DataBuffer

::: sef.core.artifacts.buffer.DataBuffer.DataSubscription

## Signals and Generic Data

::: sef.core.artifacts.Signal.Signal

::: sef.core.artifacts.data.NoData.NoData

::: sef.core.artifacts.data.CategoryData.CategoryData

::: sef.core.artifacts.data.TwoDimPointData.TwoDimPointData

::: sef.core.artifacts.data.TwoDimGraphData.TwoDimGraphData

::: sef.core.artifacts.data.VectorFieldGraphData.VectorFieldGraphData

::: sef.core.artifacts.data.TrajectoryData.TrajectoryData

## Motion and Tracking Samples

::: sef.core.artifacts.signal_sample.BoxSignalSample.BoxSignalSample

::: sef.core.artifacts.signal_sample.MultiManualSignalSample.MultiManualSignalSample

::: sef.core.artifacts.signal_sample.MultiObjectSignalSample.MultiObjectTrack

::: sef.core.artifacts.signal_sample.MultiObjectSignalSample.MultiObjectSignalSample

::: sef.core.artifacts.signal_sample.SparseOpticalFlowSignalSample.SparseOpticalFlowSignalSample

::: sef.core.artifacts.signal_sample.DenseOpticalFlowSignalSample.DenseOpticalFlowSignalSample

::: sef.core.artifacts.data.TrackingPlaybackData.TrackingPlaybackTrack

::: sef.core.artifacts.data.TrackingPlaybackData.TrackingPlaybackFrame

::: sef.core.artifacts.data.TrackingPlaybackData.TrackingPlaybackData

## ArUco Data

::: sef.core.artifacts.signal_sample.ArucoMarkerSignalSample.ArucoMarkerObservation

::: sef.core.artifacts.signal_sample.ArucoMarkerSignalSample.ArucoMarkerSignalSample

::: sef.core.artifacts.data.ArucoDisplacementData.ArucoMarkerDisplacementObservation

::: sef.core.artifacts.data.ArucoDisplacementData.ArucoMarkerDisplacementSeries

::: sef.core.artifacts.data.ArucoDisplacementData.ArucoMarkerDisplacementFrameData

::: sef.core.artifacts.data.ArucoDisplacementData.ArucoMarkerDisplacementData

::: sef.core.artifacts.data.ArucoRelativeMotionData.ArucoMarkerRelativeMotionSeries

::: sef.core.artifacts.data.ArucoRelativeMotionData.ArucoMarkerRelativeMotionData

## COCO Pose Data

::: sef.core.artifacts.signal_sample.COCOSkeletonSignalSample.COCOSkeletonSignalSample

::: sef.core.artifacts.data.COCOPoseFrameData.COCOPoseFrameData

::: sef.core.artifacts.data.COCOPoseFrameData.COCOPoseSequenceData

::: sef.core.artifacts.data.COCOPoseTennisFrameData.COCOPoseTennisFrameData

::: sef.core.artifacts.data.COCOPoseTennisFrameData.COCOPoseTennisSequenceData

## Intermediate Frame Artifacts

::: sef.core.artifacts.intermediate_frame.IntermediateFrameArtifacts.IntermediateFrameArtifactCollection

::: sef.core.artifacts.intermediate_frame.IntermediateFrameComposition.FrameComparisonPanel

::: sef.core.artifacts.mask.MaskArtifacts.MaskArtifact

::: sef.core.artifacts.mask.MaskArtifacts.MotionMaskArtifact

::: sef.core.artifacts.mask.MaskArtifacts.TargetMaskArtifact

::: sef.core.artifacts.mask.MaskArtifacts.ProtectedRegionArtifact

::: sef.core.artifacts.mask.MaskArtifacts.FrameMaskArtifact

::: sef.core.artifacts.mask.MaskArtifacts.IntermediateFrameArtifact

::: sef.core.artifacts.mask.MaskArtifacts.IntermediateFrameOverlay

