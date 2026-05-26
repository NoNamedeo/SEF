from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable

from library.core.errors import DuplicatePluginRegistrationError, InvalidPluginRegistrationError

# ── Category enum ─────────────────────────────────────────────────────────────


_PLUGIN_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DEFAULT_PLUGIN_VERSION = "1.0.0"


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidPluginRegistrationError(f"Plugin {field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise InvalidPluginRegistrationError(f"Plugin {field_name} must be a non-empty string.")
    if not _PLUGIN_IDENTIFIER_RE.fullmatch(normalized):
        raise InvalidPluginRegistrationError(
            f"Plugin {field_name} '{value}' must match {_PLUGIN_IDENTIFIER_RE.pattern}."
        )
    return normalized


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
    version: str = _DEFAULT_PLUGIN_VERSION
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _normalize_identifier(self.category, "category"))
        object.__setattr__(self, "name", _normalize_identifier(self.name, "name"))
        if not callable(self.factory):
            raise InvalidPluginRegistrationError(f"Plugin factory for '{self.name}' must be callable.")
        version = str(self.version).strip()
        if not version:
            raise InvalidPluginRegistrationError(f"Plugin '{self.name}' version must be a non-empty string.")
        object.__setattr__(self, "version", version)
        normalized_aliases = tuple(_normalize_identifier(alias, "alias") for alias in self.aliases)
        if self.name in normalized_aliases:
            raise InvalidPluginRegistrationError(f"Plugin '{self.name}' cannot use its canonical name as an alias.")
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise InvalidPluginRegistrationError(f"Plugin '{self.name}' aliases must be unique.")
        object.__setattr__(self, "aliases", normalized_aliases)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def factory_path(self) -> str:
        """Return a stable dotted path for diagnostics and exported descriptors."""
        module = getattr(self.factory, "__module__", type(self.factory).__module__)
        qualname = getattr(self.factory, "__qualname__", type(self.factory).__qualname__)
        return f"{module}.{qualname}"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable descriptor without exposing the factory object."""
        return {
            "category": self.category,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "aliases": list(self.aliases),
            "factory_path": self.factory_path,
            "metadata": dict(self.metadata),
        }


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
    Registration usually happens at startup, but the registry still protects
    reads and writes with a re-entrant lock. Snapshots returned by public
    inspection methods are immutable.
    """

    def __init__(self) -> None:
        self._definitions: dict[str, dict[str, PluginDefinition]] = {}
        self._aliases: dict[str, dict[str, str]] = {}
        self._lock = RLock()

    # ── Registration ─────────────────────────────────────────────────────────

    def register(
        self,
        category: str | PluginCategory,
        name: str,
        factory: Callable[..., Any],
        description: str = "",
        *,
        version: str = _DEFAULT_PLUGIN_VERSION,
        aliases: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> PluginDefinition:
        """
        Register a new plugin.

        Raises
        ------
        InvalidPluginRegistrationError
            If category/name/aliases/factory/version are malformed.
        DuplicatePluginRegistrationError
            If a plugin with the same (category, name) pair is already registered.
        """
        if isinstance(aliases, str):
            raise InvalidPluginRegistrationError("Plugin aliases must be an iterable of strings, not a string.")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise InvalidPluginRegistrationError("Plugin metadata must be a mapping.")

        definition = PluginDefinition(
            category=str(category),
            name=name,
            factory=factory,
            description=description,
            version=version,
            aliases=tuple(aliases),
            metadata=metadata or {},
        )
        with self._lock:
            category_map = self._definitions.setdefault(definition.category, {})
            alias_map = self._aliases.setdefault(definition.category, {})
            self._ensure_name_available(definition.category, definition.name)
            for alias in definition.aliases:
                self._ensure_name_available(definition.category, alias)
            category_map[definition.name] = definition
            for alias in definition.aliases:
                alias_map[alias] = definition.name
        return definition

    def register_definition(self, definition: PluginDefinition) -> PluginDefinition:
        """Register an already-built PluginDefinition."""
        return self.register(
            definition.category,
            definition.name,
            definition.factory,
            definition.description,
            version=definition.version,
            aliases=definition.aliases,
            metadata=definition.metadata,
        )

    def _ensure_name_available(self, category: str, name: str) -> None:
        category_map = self._definitions.get(category, {})
        alias_map = self._aliases.get(category, {})
        if name in category_map:
            raise DuplicatePluginRegistrationError(f"Plugin '{name}' already registered in category '{category}'.")
        if name in alias_map:
            canonical_name = alias_map[name]
            raise DuplicatePluginRegistrationError(
                f"Plugin alias '{name}' already points to '{canonical_name}' in category '{category}'."
            )

    # ── Lookup ───────────────────────────────────────────────────────────────

    def get(self, category: str | PluginCategory, name: str) -> PluginDefinition:
        """Return the PluginDefinition for (category, name)."""
        try:
            category_name = _normalize_identifier(str(category), "category")
            plugin_name = _normalize_identifier(name, "name")
        except InvalidPluginRegistrationError as exc:
            raise KeyError(str(exc)) from exc
        with self._lock:
            category_map = self._definitions.get(category_name, {})
            if plugin_name in category_map:
                return category_map[plugin_name]
            canonical_name = self._aliases.get(category_name, {}).get(plugin_name)
            if canonical_name and canonical_name in category_map:
                return category_map[canonical_name]
            available = self.available_names(category_name, include_aliases=True)
        raise KeyError(f"Plugin '{plugin_name}' not found in category '{category_name}'. Available: {available}")

    def create(self, category: str | PluginCategory, name: str, *args, **kwargs) -> Any:
        """Instantiate the plugin identified by (category, name)."""
        return self.get(category, name).factory(*args, **kwargs)

    def list(self, category: str | PluginCategory | None = None) -> list[PluginDefinition]:
        """Return all registered plugins, optionally filtered by category."""
        with self._lock:
            if category is None:
                return [definition for cat_map in self._definitions.values() for definition in cat_map.values()]
            return list(self._definitions.get(_normalize_identifier(str(category), "category"), {}).values())

    def contains(self, category: str | PluginCategory, name: str) -> bool:
        """Return whether a plugin name or alias is registered in a category."""
        try:
            self.get(category, name)
        except KeyError:
            return False
        return True

    def available_names(
        self,
        category: str | PluginCategory,
        *,
        include_aliases: bool = False,
    ) -> tuple[str, ...]:
        """Return registered names for a category, sorted for deterministic diagnostics."""
        category_name = _normalize_identifier(str(category), "category")
        with self._lock:
            names = set(self._definitions.get(category_name, {}).keys())
            if include_aliases:
                names.update(self._aliases.get(category_name, {}).keys())
            return tuple(sorted(names))

    def categories(self) -> tuple[str, ...]:
        """Return categories that currently contain at least one plugin."""
        with self._lock:
            return tuple(sorted(self._definitions))

    def snapshot(self) -> Mapping[str, Mapping[str, PluginDefinition]]:
        """Return an immutable category/name snapshot of canonical plugin definitions."""
        with self._lock:
            return MappingProxyType(
                {
                    category: MappingProxyType(dict(definitions))
                    for category, definitions in self._definitions.items()
                }
            )

    def describe(self, category: str | PluginCategory | None = None) -> list[dict[str, Any]]:
        """Return JSON-serializable plugin descriptors for diagnostics and UIs."""
        return [definition.as_dict() for definition in self.list(category)]


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
    from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
    from library.frame_extractors.OpenCVWebcamFrameExtractor import OpenCVWebcamFrameExtractor
    from library.frame_processors.ColorStabilizationFrameProcessor import ColorStabilizationFrameProcessor
    from library.frame_processors.DynamicObjectRemovalFrameProcessor import DynamicObjectRemovalFrameProcessor
    from library.frame_processors.OpenCVGrayFrameProcessor import OpenCVGrayFrameProcessor
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
