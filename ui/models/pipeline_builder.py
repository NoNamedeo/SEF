"""Typed models for composer state and pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

Barrier = tuple[tuple[float, float], tuple[float, float]]


class AnalysisStageKey(StrEnum):
    """Stages exposed by the UI composer."""

    FRAME_EXTRACTOR = "frame_extractor"
    FRAME_PROCESSORS = "frame_processors"
    SIGNAL_EXTRACTOR = "signal_extractor"
    SIGNAL_CLEANERS = "signal_cleaners"
    ANALYZERS = "analyzers"
    VISUALIZERS = "visualizers"


STAGE_LABELS: dict[AnalysisStageKey, str] = {
    AnalysisStageKey.FRAME_EXTRACTOR: "Frame extractor",
    AnalysisStageKey.FRAME_PROCESSORS: "Frame processors",
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
    processor_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.processor_type is not None:
            payload["processor_type"] = self.processor_type
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
class IntermediateFrameConfiguration:
    """Configuration for frame-processing debug capture and its visualizers."""

    enabled: bool = False
    max_stored_frames: int = 30
    visualizers: tuple[PluginConfig, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "max_stored_frames": self.max_stored_frames,
        }
        if self.visualizers:
            payload["visualizers"] = [item.to_dict() for item in self.visualizers]
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    """Streaming runtime settings emitted in pipeline.runtime."""

    frame_buffer_size: int = 8
    signal_buffer_size: int = 8
    data_buffer_size: int = 8
    latency_policy_name: str = "blocking"
    latency_policy_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_buffer_size": self.frame_buffer_size,
            "signal_buffer_size": self.signal_buffer_size,
            "data_buffer_size": self.data_buffer_size,
            "latency_policy": {
                "name": self.latency_policy_name,
                "params": dict(self.latency_policy_params),
            },
        }


@dataclass(frozen=True, slots=True)
class PipelineConfiguration:
    """Typed UI representation of the pipeline config passed to the builder."""

    frame_extractor: PluginConfig
    signal_extractor: PluginConfig
    frame_processors: tuple[PluginConfig, ...] = ()
    signal_cleaners: tuple[PluginConfig, ...] = ()
    analyzers: tuple[PluginConfig, ...] = ()
    visualizers: tuple[VisualizerConfig, ...] = ()
    intermediate_frames: IntermediateFrameConfiguration | None = None
    runtime: RuntimeConfiguration = field(default_factory=RuntimeConfiguration)

    def to_dict(self) -> dict[str, Any]:
        pipeline = {
            "frame_extractor": self.frame_extractor.to_dict(),
            "frame_processors": [item.to_dict() for item in self.frame_processors],
            "signal_extractor": self.signal_extractor.to_dict(),
            "signal_cleaners": [item.to_dict() for item in self.signal_cleaners],
            "analyzers": [item.to_dict() for item in self.analyzers],
            "visualizers": [item.to_dict() for item in self.visualizers],
        }
        if self.intermediate_frames is not None:
            pipeline["intermediate_frames"] = self.intermediate_frames.to_dict()
        pipeline["runtime"] = self.runtime.to_dict()
        return {"pipeline": pipeline}


@dataclass(frozen=True, slots=True)
class BuilderStateSnapshot:
    """Stable snapshot of the Streamlit composer state."""

    mode: str
    frame_extractor: str
    frame_processors: tuple[str, ...]
    signal_extractor: str
    signal_cleaners: tuple[str, ...]
    analyzers: tuple[str, ...]
    visualizers: tuple[str, ...]
    intermediate_visualizers: tuple[str, ...]
    visualizer_targets: dict[str, str]
    resize_label: str
    stride: int
    max_frames_enabled: bool
    max_frames: int
    webcam_index: int
    tracker: str
    show_windows: bool
    moving_average_window: int
    outlier_threshold: float
    outlier_mode: str
    widener: float
    aruco_quality_threshold: float
    aruco_alpha_high_quality: float
    aruco_alpha_low_quality: float
    aruco_max_jump_px: float
    aruco_smooth_corners: bool
    smoothing_alpha: float
    smoothing_reset: float
    background_method: str
    background_shadows: bool
    multi_max_objects: int
    multi_similarity: float
    dense_cell_size: int
    barrier_names: tuple[str, ...]
    branching_rules: tuple[str, ...]
    color_stab_color_space: str = "LAB"
    color_stab_techniques: tuple[str, ...] = ("luminance_normalization", "temporal_smoothing")
    color_stab_strength: float = 0.85
    color_stab_temporal_alpha: float = 0.92
    color_stab_chroma: bool = True
    color_stab_chroma_strength: float = 0.20
    color_stab_hist_min_std: float = 4.0
    color_stab_hist_max_gain: float = 1.35
    color_stab_lum_max_shift: float = 48.0
    color_stab_gamma: float = 0.0
    color_stab_clahe_clip: float = 2.0
    color_stab_clahe_strength: float = 0.35
    color_stab_emit_metrics: bool = True
    color_stab_emit_overlay: bool = False
    color_stab_emit_intermediate: bool = False
    dynamic_removal_sampling_stride: int = 5
    dynamic_removal_max_sampled_frames: int = 60
    dynamic_removal_difference_threshold: int = 35
    dynamic_removal_morph_kernel_size: int = 5
    dynamic_removal_opening_iterations: int = 1
    dynamic_removal_closing_iterations: int = 2
    dynamic_removal_dilation_iterations: int = 1
    dynamic_removal_min_component_area: int = 80
    dynamic_removal_max_processed_frames: int = 300
    dynamic_removal_emit_intermediate: bool = False
    intermediate_capture_enabled: bool = False
    intermediate_capture_max_frames: int = 30
    runtime_frame_buffer_size: int = 8
    runtime_signal_buffer_size: int = 8
    runtime_data_buffer_size: int = 8
    runtime_latency_policy: str = "blocking"
    runtime_adaptive_min_interval: int = 1
    runtime_adaptive_max_interval: int = 8
    runtime_adaptive_low_watermark: float = 0.25
    runtime_adaptive_high_watermark: float = 0.75

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
