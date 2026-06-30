from __future__ import annotations

from sef.builtin._optional_dependencies import lazy_component_factory
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
    from sef.builtin.signal_cleaners.ArUco.ArucoTemporalStabilizerCleaner import ArucoTemporalStabilizerCleaner
    from sef.builtin.signal_cleaners.single_tracker.MovingAverageCleaner import MovingAverageCleaner

    registry = PluginRegistry()

    registry.register(
        PluginCategory.FRAME_EXTRACTOR,
        "opencv_buffered",
        lazy_component_factory(
            "sef.builtin.frame_extractors.OpenCVBufferedFrameExtractor.OpenCVBufferedFrameExtractor",
            extra="opencv",
        ),
        "Extract frames from a video using OpenCV.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.FRAME_EXTRACTOR,
        "opencv_webcam",
        lazy_component_factory(
            "sef.builtin.frame_extractors.OpenCVWebcamFrameExtractor.OpenCVWebcamFrameExtractor",
            extra="opencv",
        ),
        "Capture frames from a local webcam using OpenCV.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.SINGLE_FRAME_PROCESSOR,
        "opencv_gray",
        lazy_component_factory(
            "sef.builtin.frame_processors.OpenCV.OpenCVGrayFrameProcessor.OpenCVGrayFrameProcessor",
            extra="opencv",
        ),
        "Convert frames to grayscale.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.SINGLE_FRAME_PROCESSOR,
        "color_stabilization",
        lazy_component_factory(
            "sef.builtin.frame_processors.ColorStabilizationFrameProcessor.ColorStabilizationFrameProcessor",
            extra="opencv",
        ),
        "Stabilize illumination, brightness, and chromatic drift between frames.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.SINGLE_FRAME_PROCESSOR,
        "smoothing",
        lazy_component_factory(
            "sef.builtin.frame_processors.SmoothingFrameProcessor.SmoothingFrameProcessor",
            extra="opencv",
        ),
        "Apply temporal smoothing between consecutive frames.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.FRAME_BUFFER_PROCESSOR,
        "dynamic_object_removal",
        lazy_component_factory(
            "sef.builtin.frame_processors.DynamicObjectRemovalFrameProcessor.DynamicObjectRemovalFrameProcessor",
            extra="opencv",
        ),
        "Remove transient dynamic objects using a temporal median background.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.FRAME_BUFFER_PROCESSOR,
        "motion_magnification",
        lazy_component_factory(
            "sef.builtin.frame_processors.motion_magnification.PhaseMagnificationFrameProcessor.PhaseMagnificationFrameProcessor",
            extra="opencv",
        ),
        "Magnify subtle motions by wrapping the external phase-based MATLAB pipeline.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.SIGNAL_EXTRACTOR,
        "opencv_tracker",
        lazy_component_factory(
            "sef.builtin.signal_extractors.OpenCVBufferedSignalExtractor.OpenCVBufferedSignalExtractor",
            extra="opencv",
        ),
        "Track a single ROI with an OpenCV tracker.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.SIGNAL_EXTRACTOR,
        "aruco_marker",
        lazy_component_factory(
            "sef.builtin.signal_extractors.ArucoMarkerSignalExtractor.ArucoMarkerSignalExtractor",
            extra="opencv",
        ),
        "Detect configurable ArUco markers frame by frame.",
        metadata={"optional_extra": "opencv"},
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

    registry.register(
        PluginCategory.VISUALIZER,
        "matplotlib",
        lazy_component_factory(
            "sef.builtin.visualizers.Matplotlib.MatplotlibFunctionVisualizer.MatplotlibFunctionVisualizer",
            extra="visualization",
        ),
        "Plot analytical data with Matplotlib.",
        metadata={"optional_extra": "visualization"},
    )
    registry.register(
        PluginCategory.VISUALIZER,
        "aruco_motion_plot",
        lazy_component_factory(
            "sef.builtin.visualizers.Matplotlib.MatplotlibArucoMotionVisualizer.MatplotlibArucoMotionVisualizer",
            extra="visualization",
        ),
        "Render ArUco displacement and relative-motion plots.",
        metadata={"optional_extra": "visualization"},
    )

    registry.register(
        PluginCategory.VISUALIZER,
        "aruco_annotated_video",
        lazy_component_factory(
            "sef.builtin.visualizers.ArUco.ArucoAnnotatedVideoVisualizer.ArucoAnnotatedVideoVisualizer",
            extra="opencv",
        ),
        "Render annotated MP4 output for ArUco detections.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.VISUALIZER,
        "intermediate_frames",
        lazy_component_factory(
            "sef.builtin.visualizers.intermediate_frames.IntermediateFramesVisualizer.IntermediateFramesVisualizer",
            extra="opencv",
        ),
        "Render each captured preprocessing snapshot as a comparison PNG.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.VISUALIZER,
        "intermediate_frames_grid",
        lazy_component_factory(
            "sef.builtin.visualizers.intermediate_frames.IntermediateFramesGridVisualizer.IntermediateFramesGridVisualizer",
            extra="opencv",
        ),
        "Render captured preprocessing snapshots as a bounded comparison grid.",
        metadata={"optional_extra": "opencv"},
    )
    registry.register(
        PluginCategory.BRANCHING_RULE,
        "default_track_branch",
        lazy_component_factory(
            "sef.builtin.branching_rules.NewTrackBranchingRule.NewTrackBranchingRule",
            extra="opencv",
        ),
        "Branch once when the primary multi-object tracker creates its seed track.",
        metadata={"optional_extra": "opencv"},
    )

    return registry
