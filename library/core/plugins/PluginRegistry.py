from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

# ── Category enum ─────────────────────────────────────────────────────────────


class PluginCategory(StrEnum):
    """
    Canonical category identifiers for all pluggable pipeline components.

    Design rationale
    ----------------
    Replacing raw strings with a StrEnum eliminates typo-induced KeyErrors
    at plugin registration and lookup time, provides IDE autocompletion,
    and makes mypy able to catch category mismatches statically.

    StrEnum (Python 3.11+) means each member compares equal to its string
    value, so existing code that uses the string literal (e.g. "analyzer")
    continues to work without modification during migration.
    """

    FRAME_EXTRACTOR = "frame_extractor"
    SINGLE_FRAME_PROCESSOR = "single_frame_processor"
    FRAME_BUFFER_PROCESSOR = "frame_buffer_processor"
    SIGNAL_EXTRACTOR = "signal_extractor"
    SIGNAL_CLEANER = "signal_cleaner"
    ANALYZER = "analyzer"
    VISUALIZER = "visualizer"
    BRANCHING_RULE = "branching_rule"


# ── Plugin definition ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PluginDefinition:
    """
    Immutable descriptor for a registered plugin.

    frozen=True  prevents accidental mutation after registration.
    slots=True   reduces memory overhead for registries with many plugins.
    """

    category: str
    name: str
    factory: Callable[..., Any]
    description: str = ""


# ── Registry ──────────────────────────────────────────────────────────────────


class PluginRegistry:
    """
    Lightweight, centralised registry for pluggable pipeline components.

    Design rationale
    ----------------
    The registry is the single source of truth for available components.
    It decouples builders (which only know category + name) from concrete
    implementations (which only need to call register() once).

    Thread safety
    -------------
    Registration is expected to happen at startup (single-threaded).
    Concurrent reads are safe; concurrent writes are not protected by a
    lock and should be avoided at runtime.
    """

    def __init__(self) -> None:
        self._definitions: dict[str, dict[str, PluginDefinition]] = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def register(
        self,
        category: str | PluginCategory,
        name: str,
        factory: Callable[..., Any],
        description: str = "",
    ) -> PluginDefinition:
        """
        Register a new plugin.

        Raises
        ------
        ValueError
            If a plugin with the same (category, name) pair is already registered.
        """
        category = str(category)
        category_map = self._definitions.setdefault(category, {})
        if name in category_map:
            raise ValueError(f"Plugin '{name}' already registered in category '{category}'.")
        definition = PluginDefinition(
            category=category,
            name=name,
            factory=factory,
            description=description,
        )
        category_map[name] = definition
        return definition

    # ── Lookup ───────────────────────────────────────────────────────────────

    def get(self, category: str | PluginCategory, name: str) -> PluginDefinition:
        """Return the PluginDefinition for (category, name)."""
        category = str(category)
        try:
            return self._definitions[category][name]
        except KeyError as exc:
            available = list(self._definitions.get(category, {}).keys())
            raise KeyError(f"Plugin '{name}' not found in category '{category}'. Available: {available}") from exc

    def create(self, category: str | PluginCategory, name: str, *args, **kwargs) -> Any:
        """Instantiate the plugin identified by (category, name)."""
        return self.get(category, name).factory(*args, **kwargs)

    def list(self, category: str | PluginCategory | None = None) -> list[PluginDefinition]:
        """Return all registered plugins, optionally filtered by category."""
        if category is None:
            return [definition for cat_map in self._definitions.values() for definition in cat_map.values()]
        return list(self._definitions.get(str(category), {}).values())


# ── Built-in registry factory ─────────────────────────────────────────────────


def create_builtin_registry() -> PluginRegistry:
    """
    Register the built-in OpenCV and Matplotlib components and return the registry.

    This factory is the canonical entry-point for production use.
    Custom plugins can be added to the returned registry before passing
    it to a builder.

    Example
    -------
    >>> registry = create_builtin_registry()
    >>> registry.register(PluginCategory.ANALYZER, "my_analyzer", MyAnalyzer)
    >>> builder = ConfigPipelineBuilder(registry)
    """
    from library.analyzers.ArucoMarkerDisplacementAnalyzer import ArucoMarkerDisplacementAnalyzer
    from library.analyzers.ArucoMarkerRelativeMotionAnalyzer import ArucoMarkerRelativeMotionAnalyzer
    from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
    from library.branching_rules.NewTrackBranchingRule import NewTrackBranchingRule
    from library.frame_processors.ColorStabilizationFrameProcessor import ColorStabilizationFrameProcessor
    from library.frame_processors.DynamicObjectRemovalFrameProcessor import DynamicObjectRemovalFrameProcessor
    from library.frame_processors.OpenCVGrayFrameProcessor import OpenCVGrayFrameProcessor
    from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
    from library.frame_extractors.OpenCVWebcamFrameExtractor import OpenCVWebcamFrameExtractor
    from library.signal_cleaners.ArucoTemporalStabilizerCleaner import ArucoTemporalStabilizerCleaner
    from library.signal_cleaners.MovingAverageCleaner import MovingAverageCleaner
    from library.signal_extractors.ArucoMarkerSignalExtractor import ArucoMarkerSignalExtractor
    from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
    from library.visualizers.ArucoAnnotatedVideoVisualizer import ArucoAnnotatedVideoVisualizer
    from library.visualizers.IntermediateFramesGridVisualizer import IntermediateFramesGridVisualizer
    from library.visualizers.IntermediateFramesVisualizer import IntermediateFramesVisualizer
    from library.visualizers.MatplotlibArucoMotionVisualizer import MatplotlibArucoMotionVisualizer
    from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

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

    registry.register(
        PluginCategory.VISUALIZER,
        "matplotlib",
        MatplotlibFunctionVisualizer,
        "Plot analytical data with Matplotlib.",
    )

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
