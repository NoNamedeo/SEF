from __future__ import annotations

from sef.core.plugins.PluginRegistry import PluginCategory, PluginRegistry


def create_builtin_registry() -> PluginRegistry:
    """
    Register the built-in OpenCV, Matplotlib, and SEF concrete components.

    This adapter lives in ``sef.builtin`` so the core plugin registry remains a
    pure registry abstraction. Application code that wants built-in components
    should call this factory directly or use ``sef.default_registry()``.
    """
    from sef.builtin.analyzers.ArUco.ArucoMarkerDisplacementAnalyzer import ArucoMarkerDisplacementAnalyzer
    from sef.builtin.analyzers.ArUco.ArucoMarkerRelativeMotionAnalyzer import ArucoMarkerRelativeMotionAnalyzer
    from sef.builtin.analyzers.single_tracker.VerticalPositionAnalyzer import VerticalPositionAnalyzer
    from sef.builtin.branching_rules.NewTrackBranchingRule import NewTrackBranchingRule
    from sef.builtin.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
    from sef.builtin.frame_extractors.OpenCVWebcamFrameExtractor import OpenCVWebcamFrameExtractor
    from sef.builtin.frame_processors.ColorStabilizationFrameProcessor import ColorStabilizationFrameProcessor
    from sef.builtin.frame_processors.DynamicObjectRemovalFrameProcessor import DynamicObjectRemovalFrameProcessor
    from sef.builtin.frame_processors.OpenCV.OpenCVGrayFrameProcessor import OpenCVGrayFrameProcessor
    from sef.builtin.frame_processors.motion_magnification.PhaseMagnificationFrameProcessor import PhaseMagnificationFrameProcessor
    from sef.builtin.signal_cleaners.ArUco.ArucoTemporalStabilizerCleaner import ArucoTemporalStabilizerCleaner
    from sef.builtin.signal_cleaners.single_tracker.MovingAverageCleaner import MovingAverageCleaner
    from sef.builtin.signal_extractors.ArucoMarkerSignalExtractor import ArucoMarkerSignalExtractor
    from sef.builtin.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
    from sef.builtin.visualizers.ArUco.ArucoAnnotatedVideoVisualizer import ArucoAnnotatedVideoVisualizer
    from sef.builtin.visualizers.intermediate_frames.IntermediateFramesGridVisualizer import IntermediateFramesGridVisualizer
    from sef.builtin.visualizers.intermediate_frames.IntermediateFramesVisualizer import IntermediateFramesVisualizer
    from sef.builtin.visualizers.Matplotlib.MatplotlibArucoMotionVisualizer import MatplotlibArucoMotionVisualizer
    from sef.builtin.visualizers.Matplotlib.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

    registry = PluginRegistry()

    registry.register(
        PluginCategory.FRAME_EXTRACTOR,
        "opencv_buffered",
        OpenCVBufferedFrameExtractor,
        "Extract frames from a video using OpenCV.",
    )
    registry.register(
        PluginCategory.FRAME_EXTRACTOR,
        "opencv_webcam",
        OpenCVWebcamFrameExtractor,
        "Capture frames from a local webcam using OpenCV.",
    )
    registry.register(
        PluginCategory.SINGLE_FRAME_PROCESSOR,
        "opencv_gray",
        OpenCVGrayFrameProcessor,
        "Convert frames to grayscale.",
    )
    registry.register(
        PluginCategory.SINGLE_FRAME_PROCESSOR,
        "color_stabilization",
        ColorStabilizationFrameProcessor,
        "Stabilize illumination, brightness, and chromatic drift between frames.",
    )
    registry.register(
        PluginCategory.FRAME_BUFFER_PROCESSOR,
        "dynamic_object_removal",
        DynamicObjectRemovalFrameProcessor,
        "Remove transient dynamic objects using a temporal median background.",
    )
    registry.register(
        PluginCategory.FRAME_BUFFER_PROCESSOR,
        "motion_magnification",
        PhaseMagnificationFrameProcessor,
        "Magnify subtle motions by wrapping the external phase-based MATLAB pipeline.",
    )
    registry.register(
        PluginCategory.SIGNAL_EXTRACTOR,
        "opencv_tracker",
        OpenCVBufferedSignalExtractor,
        "Track a single ROI with an OpenCV tracker.",
    )
    registry.register(
        PluginCategory.SIGNAL_EXTRACTOR,
        "aruco_marker",
        ArucoMarkerSignalExtractor,
        "Detect configurable ArUco markers frame by frame.",
    )
    registry.register(
        PluginCategory.SIGNAL_CLEANER,
        "moving_average",
        MovingAverageCleaner,
        "Smooth centroid coordinates with a moving average.",
    )
    registry.register(
        PluginCategory.SIGNAL_CLEANER,
        "aruco_temporal_stabilizer",
        ArucoTemporalStabilizerCleaner,
        "Stabilize ArUco marker centers and corners over time using quality-aware temporal smoothing.",
    )
    registry.register(
        PluginCategory.ANALYZER,
        "vertical_position",
        VerticalPositionAnalyzer,
        "Extract the vertical position series from tracked centroids.",
    )
    registry.register(
        PluginCategory.ANALYZER,
        "aruco_displacement",
        ArucoMarkerDisplacementAnalyzer,
        "Compute per-marker 2D displacement over time.",
    )
    registry.register(
        PluginCategory.ANALYZER,
        "aruco_relative_motion",
        ArucoMarkerRelativeMotionAnalyzer,
        "Measure relative distance changes between ArUco marker pairs.",
    )

    if MatplotlibFunctionVisualizer is not None:
        registry.register(
            PluginCategory.VISUALIZER,
            "matplotlib",
            MatplotlibFunctionVisualizer,
            "Plot analytical data with Matplotlib.",
        )
    if MatplotlibArucoMotionVisualizer is not None:
        registry.register(
            PluginCategory.VISUALIZER,
            "aruco_motion_plot",
            MatplotlibArucoMotionVisualizer,
            "Render ArUco displacement and relative-motion plots.",
        )

    registry.register(
        PluginCategory.VISUALIZER,
        "aruco_annotated_video",
        ArucoAnnotatedVideoVisualizer,
        "Render annotated MP4 output for ArUco detections.",
    )
    registry.register(
        PluginCategory.VISUALIZER,
        "intermediate_frames",
        IntermediateFramesVisualizer,
        "Render each captured preprocessing snapshot as a comparison PNG.",
    )
    registry.register(
        PluginCategory.VISUALIZER,
        "intermediate_frames_grid",
        IntermediateFramesGridVisualizer,
        "Render captured preprocessing snapshots as a bounded comparison grid.",
    )
    registry.register(
        PluginCategory.BRANCHING_RULE,
        "default_track_branch",
        NewTrackBranchingRule,
        "Branch once when the primary multi-object tracker creates its seed track.",
    )

    return registry
