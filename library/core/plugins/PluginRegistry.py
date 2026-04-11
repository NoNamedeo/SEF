from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class PluginDefinition:
    category: str
    name: str
    factory: Callable[..., Any]
    description: str = ""


class PluginRegistry:
    """Lightweight registry for pluggable pipeline components."""

    def __init__(self):
        self._definitions: dict[str, dict[str, PluginDefinition]] = {}

    def register(
        self,
        category: str,
        name: str,
        factory: Callable[..., Any],
        description: str = "",
    ) -> PluginDefinition:
        category_map = self._definitions.setdefault(category, {})
        if name in category_map:
            raise ValueError(f"Plugin '{name}' already registered in category '{category}'")

        definition = PluginDefinition(
            category=category,
            name=name,
            factory=factory,
            description=description,
        )
        category_map[name] = definition
        return definition

    def get(self, category: str, name: str) -> PluginDefinition:
        try:
            return self._definitions[category][name]
        except KeyError as exc:
            raise KeyError(f"Plugin '{name}' not found in category '{category}'") from exc

    def create(self, category: str, name: str, *args, **kwargs):
        return self.get(category, name).factory(*args, **kwargs)

    def list(self, category: str | None = None) -> list[PluginDefinition]:
        if category is None:
            return [
                definition
                for category_map in self._definitions.values()
                for definition in category_map.values()
            ]
        return list(self._definitions.get(category, {}).values())


def create_builtin_registry() -> PluginRegistry:
    """Register the built-in OpenCV and Matplotlib components."""
    from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
    from library.frame_cleaners.OpenCVGrayFrameCleaner import OpenCVGrayFrameCleaner
    from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
    from library.signal_cleaners.MovingAverageCleaner import OpenCVMovingAverageCleaner
    from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
    from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

    registry = PluginRegistry()
    registry.register(
        "frame_extractor",
        "opencv_buffered",
        OpenCVBufferedFrameExtractor,
        "Extract frames from a video using OpenCV.",
    )
    registry.register(
        "frame_cleaner",
        "opencv_gray",
        OpenCVGrayFrameCleaner,
        "Convert frames to grayscale.",
    )
    registry.register(
        "signal_extractor",
        "opencv_tracker",
        OpenCVBufferedSignalExtractor,
        "Track a single ROI with an OpenCV tracker.",
    )
    registry.register(
        "signal_cleaner",
        "moving_average",
        OpenCVMovingAverageCleaner,
        "Smooth centroid coordinates with a moving average.",
    )
    registry.register(
        "analyzer",
        "vertical_position",
        VerticalPositionAnalyzer,
        "Extract the vertical position series from tracked centroids.",
    )
    registry.register(
        "visualizer",
        "matplotlib",
        MatplotlibFunctionVisualizer,
        "Plot analytical data with Matplotlib.",
    )
    return registry
