"""Typed models for composer state and pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

Barrier = tuple[tuple[float, float], tuple[float, float]]


class AnalysisStageKey(StrEnum):
    """Stages exposed by the UI composer."""

    FRAME_EXTRACTOR = "frame_extractor"
    FRAME_CLEANERS = "frame_cleaners"
    SIGNAL_EXTRACTOR = "signal_extractor"
    SIGNAL_CLEANERS = "signal_cleaners"
    ANALYZERS = "analyzers"
    VISUALIZERS = "visualizers"


STAGE_LABELS: dict[AnalysisStageKey, str] = {
    AnalysisStageKey.FRAME_EXTRACTOR: "Frame extractor",
    AnalysisStageKey.FRAME_CLEANERS: "Frame cleaners",
    AnalysisStageKey.SIGNAL_EXTRACTOR: "Signal extractor",
    AnalysisStageKey.SIGNAL_CLEANERS: "Signal cleaners",
    AnalysisStageKey.ANALYZERS: "Analyzers",
    AnalysisStageKey.VISUALIZERS: "Visualizers",
}


@dataclass(frozen=True, slots=True)
class PluginConfig:
    """Single plugin instance configuration."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.params:
            payload["params"] = self.params
        return payload


@dataclass(frozen=True, slots=True)
class VisualizerConfig:
    """Visualizer configuration including optional result binding."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    result_indices: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.params:
            payload["params"] = self.params
        if self.result_indices is not None:
            payload["result_indices"] = list(self.result_indices)
        return payload


@dataclass(frozen=True, slots=True)
class PipelineConfiguration:
    """Typed UI representation of the pipeline config passed to the builder."""

    frame_extractor: PluginConfig
    signal_extractor: PluginConfig
    frame_cleaners: tuple[PluginConfig, ...] = ()
    signal_cleaners: tuple[PluginConfig, ...] = ()
    analyzers: tuple[PluginConfig, ...] = ()
    visualizers: tuple[VisualizerConfig, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": {
                "frame_extractor": self.frame_extractor.to_dict(),
                "frame_cleaners": [item.to_dict() for item in self.frame_cleaners],
                "signal_extractor": self.signal_extractor.to_dict(),
                "signal_cleaners": [item.to_dict() for item in self.signal_cleaners],
                "analyzers": [item.to_dict() for item in self.analyzers],
                "visualizers": [item.to_dict() for item in self.visualizers],
            }
        }


@dataclass(frozen=True, slots=True)
class BuilderStateSnapshot:
    """Stable snapshot of the Streamlit composer state."""

    mode: str
    frame_extractor: str
    frame_cleaners: tuple[str, ...]
    signal_extractor: str
    signal_cleaners: tuple[str, ...]
    analyzers: tuple[str, ...]
    visualizers: tuple[str, ...]
    visualizer_targets: dict[str, str]
    resize_label: str
    stride: int
    max_frames_enabled: bool
    max_frames: int
    tracker: str
    show_windows: bool
    moving_average_window: int
    outlier_threshold: float
    outlier_mode: str
    widener: float
    smoothing_alpha: float
    smoothing_reset: float
    background_method: str
    background_shadows: bool
    multi_max_objects: int
    multi_similarity: float
    dense_cell_size: int
    barrier_names: tuple[str, ...]
    branching_rules: tuple[str, ...]

    @property
    def resize(self) -> tuple[int, int] | None:
        if self.resize_label == "Originale":
            return None
        width, height = self.resize_label.split("x")
        return int(width), int(height)


@dataclass(frozen=True, slots=True)
class BarrierSelectionState:
    """Explicit barrier-selection progress stored by the UI."""

    names: tuple[str, ...]
    confirmed: tuple[tuple[str, Barrier], ...] = ()
    next_index: int = 0

    @property
    def complete(self) -> bool:
        return bool(self.names) and self.next_index >= len(self.names)

    @property
    def current_name(self) -> str | None:
        if self.complete or self.next_index >= len(self.names):
            return None
        return self.names[self.next_index]

    def as_dict(self) -> dict[str, Barrier]:
        return {name: barrier for name, barrier in self.confirmed}
