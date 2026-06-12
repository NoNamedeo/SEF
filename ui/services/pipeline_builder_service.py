"""State and configuration services for the Streamlit composer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from sef.core.plugins.PluginRegistry import PluginCategory
from ui.models.pipeline_builder import (
    STAGE_LABELS,
    AnalysisStageKey,
    BarrierSelectionState,
    BuilderStateSnapshot,
    IntermediateFrameConfiguration,
    PipelineConfiguration,
    PluginConfig,
    RuntimeConfiguration,
    VisualizerConfig,
)
from ui.services.realtime_preview_service import STREAMLIT_COCO_POSE_REALTIME_VISUALIZER, STREAMLIT_REALTIME_VISUALIZERS
from ui.state import session

TRACKING_ANALYZERS = (
    "vertical_position",
    "horizontal_position",
    "vertical_velocity",
    "horizontal_velocity",
    "vertical_frequency",
    "horizontal_frequency",
)

TRACKING_SIGNAL_EXTRACTORS = {"opencv_tracker", "opencv_stream_tracker"}
YOLO_POSE_SIGNAL_EXTRACTOR = "yolo_coco_pose"
YOLO_POSE_ANALYZERS = ("coco_pose_stream",)
YOLO_POSE_VISUALIZERS = {
    "opencv_coco_pose_realtime",
    "opencv_coco_tennis_pose_realtime",
}

ARUCO_ANALYZERS = (
    "aruco_displacement",
    "aruco_relative_motion",
)

INTERMEDIATE_FRAME_VISUALIZERS = frozenset(
    {
        "intermediate_frames",
        "intermediate_frames_grid",
    }
)

FRAME_BUFFER_PROCESSORS = frozenset({"dynamic_object_removal", "motion_magnification"})

BUILDER_LAST_SYNCED_MODE_KEY = "sef_builder_last_synced_mode"

STAGE_EDIT_OPTIONS = tuple(stage.value for stage in AnalysisStageKey)


def initialise_builder_state(registry) -> None:
    """Populate the Streamlit session with explicit builder defaults."""
    selected_stage = st.query_params.get("stage", AnalysisStageKey.FRAME_EXTRACTOR.value)
    defaults = {
        "sef_builder_frame_extractor": first_plugin_name(registry, PluginCategory.FRAME_EXTRACTOR),
        "sef_builder_frame_processors": ["smoothing"],
        "sef_builder_signal_extractor": "opencv_tracker",
        "sef_builder_signal_cleaners": ["moving_average"],
        "sef_builder_analyzers": ["vertical_position"],
        "sef_builder_visualizers": [],
        "sef_builder_intermediate_visualizers": [],
        "sef_builder_visualizer_targets": {},
        "sef_builder_mode": "Single object tracking",
        "sef_builder_resize": "640x480",
        "sef_builder_stride": 2,
        "sef_builder_max_frames_enabled": True,
        "sef_builder_max_frames": 180,
        "sef_builder_webcam_index": 0,
        "sef_builder_webcam_mirror": False,
        "sef_builder_webcam_width": 0,
        "sef_builder_webcam_height": 0,
        "sef_builder_webcam_fps": 0,
        "sef_builder_tracker": "MIL",
        "sef_builder_show_windows": False,
        "sef_builder_yolo_model_name": "yolo11s-pose.pt",
        "sef_builder_yolo_include_frame_image": True,
        "sef_builder_yolo_draw_source_frame": True,
        "sef_builder_yolo_keypoint_threshold": 0.30,
        "sef_builder_coco_pose_retain_frames": False,
        "sef_builder_coco_pose_include_normalized": False,
        "sef_builder_coco_norm_center_on_pelvis": True,
        "sef_builder_coco_norm_normalize_scale": True,
        "sef_builder_coco_norm_align_rotation": False,
        "sef_builder_coco_norm_min_scale": 0.000001,
        "sef_builder_mavg_window": 5,
        "sef_builder_outlier_threshold": 3.5,
        "sef_builder_outlier_mode": "clip",
        "sef_builder_widener": 1.4,
        "sef_builder_aruco_quality_threshold": 0.45,
        "sef_builder_aruco_alpha_high_quality": 0.65,
        "sef_builder_aruco_alpha_low_quality": 0.20,
        "sef_builder_aruco_max_jump_px": 2.0,
        "sef_builder_aruco_smooth_corners": True,
        "sef_builder_smoothing_alpha": 0.86,
        "sef_builder_smoothing_reset": 65.0,
        "sef_builder_bg_method": "MOG2",
        "sef_builder_bg_shadows": False,
        "sef_builder_multi_max_objects": 4,
        "sef_builder_multi_similarity": 0.62,
        "sef_builder_dense_cell_size": 16,
        "sef_builder_barrier_names": "A, B",
        "sef_builder_branching_rules": [],
        "sef_builder_color_stab_color_space": "LAB",
        "sef_builder_color_stab_techniques": ["luminance_normalization", "temporal_smoothing"],
        "sef_builder_color_stab_strength": 0.85,
        "sef_builder_color_stab_temporal_alpha": 0.92,
        "sef_builder_color_stab_chroma": True,
        "sef_builder_color_stab_chroma_strength": 0.20,
        "sef_builder_color_stab_hist_min_std": 4.0,
        "sef_builder_color_stab_hist_max_gain": 1.35,
        "sef_builder_color_stab_lum_max_shift": 48.0,
        "sef_builder_color_stab_gamma": 0.0,
        "sef_builder_color_stab_clahe_clip": 2.0,
        "sef_builder_color_stab_clahe_strength": 0.35,
        "sef_builder_color_stab_emit_metrics": True,
        "sef_builder_color_stab_emit_overlay": False,
        "sef_builder_color_stab_emit_intermediate": False,
        "sef_builder_dynamic_removal_sampling_stride": 5,
        "sef_builder_dynamic_removal_max_sampled_frames": 60,
        "sef_builder_dynamic_removal_difference_threshold": 35,
        "sef_builder_dynamic_removal_morph_kernel_size": 5,
        "sef_builder_dynamic_removal_opening_iterations": 1,
        "sef_builder_dynamic_removal_closing_iterations": 2,
        "sef_builder_dynamic_removal_dilation_iterations": 1,
        "sef_builder_dynamic_removal_min_component_area": 80,
        "sef_builder_dynamic_removal_max_processed_frames": 300,
        "sef_builder_dynamic_removal_emit_intermediate": False,
        "sef_builder_intermediate_capture_enabled": False,
        "sef_builder_intermediate_capture_max_frames": 30,
        "sef_builder_runtime_frame_buffer_size": 8,
        "sef_builder_runtime_signal_buffer_size": 8,
        "sef_builder_runtime_data_buffer_size": 8,
        "sef_builder_runtime_latency_policy": "blocking",
        "sef_builder_runtime_adaptive_min_interval": 1,
        "sef_builder_runtime_adaptive_max_interval": 8,
        "sef_builder_runtime_adaptive_low_watermark": 0.25,
        "sef_builder_runtime_adaptive_high_watermark": 0.75,
        "sef_selected_stage": selected_stage if selected_stage in STAGE_EDIT_OPTIONS else AnalysisStageKey.FRAME_EXTRACTOR.value,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault(
        BUILDER_LAST_SYNCED_MODE_KEY,
        st.session_state.get("sef_builder_mode", "Single object tracking"),
    )


def builder_state() -> BuilderStateSnapshot:
    """Return a typed snapshot of the current composer state."""
    return BuilderStateSnapshot(
        mode=str(st.session_state.get("sef_builder_mode", "Single object tracking")),
        frame_extractor=str(st.session_state.get("sef_builder_frame_extractor", "")),
        frame_processors=tuple(st.session_state.get("sef_builder_frame_processors", [])),
        signal_extractor=str(st.session_state.get("sef_builder_signal_extractor", "opencv_tracker")),
        signal_cleaners=tuple(st.session_state.get("sef_builder_signal_cleaners", [])),
        analyzers=tuple(st.session_state.get("sef_builder_analyzers", [])),
        visualizers=tuple(st.session_state.get("sef_builder_visualizers", [])),
        intermediate_visualizers=tuple(st.session_state.get("sef_builder_intermediate_visualizers", [])),
        visualizer_targets=dict(st.session_state.get("sef_builder_visualizer_targets", {})),
        resize_label=str(st.session_state.get("sef_builder_resize", "640x480")),
        stride=int(st.session_state.get("sef_builder_stride", 2)),
        max_frames_enabled=bool(st.session_state.get("sef_builder_max_frames_enabled", True)),
        max_frames=int(st.session_state.get("sef_builder_max_frames", 180)),
        webcam_index=int(st.session_state.get("sef_builder_webcam_index", 0)),
        webcam_mirror=bool(st.session_state.get("sef_builder_webcam_mirror", False)),
        webcam_width=int(st.session_state.get("sef_builder_webcam_width", 0)),
        webcam_height=int(st.session_state.get("sef_builder_webcam_height", 0)),
        webcam_fps=int(st.session_state.get("sef_builder_webcam_fps", 0)),
        tracker=str(st.session_state.get("sef_builder_tracker", "MIL")),
        show_windows=bool(st.session_state.get("sef_builder_show_windows", False)),
        yolo_model_name=str(st.session_state.get("sef_builder_yolo_model_name", "yolo11s-pose.pt")),
        yolo_include_frame_image=bool(st.session_state.get("sef_builder_yolo_include_frame_image", True)),
        yolo_draw_source_frame=bool(st.session_state.get("sef_builder_yolo_draw_source_frame", True)),
        yolo_keypoint_threshold=float(st.session_state.get("sef_builder_yolo_keypoint_threshold", 0.30)),
        coco_pose_retain_frames=bool(st.session_state.get("sef_builder_coco_pose_retain_frames", False)),
        coco_pose_include_normalized_skeleton=bool(st.session_state.get("sef_builder_coco_pose_include_normalized", False)),
        coco_norm_center_on_pelvis=bool(st.session_state.get("sef_builder_coco_norm_center_on_pelvis", True)),
        coco_norm_normalize_scale=bool(st.session_state.get("sef_builder_coco_norm_normalize_scale", True)),
        coco_norm_align_rotation=bool(st.session_state.get("sef_builder_coco_norm_align_rotation", False)),
        coco_norm_min_scale=float(st.session_state.get("sef_builder_coco_norm_min_scale", 0.000001)),
        moving_average_window=int(st.session_state.get("sef_builder_mavg_window", 5)),
        outlier_threshold=float(st.session_state.get("sef_builder_outlier_threshold", 3.5)),
        outlier_mode=str(st.session_state.get("sef_builder_outlier_mode", "clip")),
        widener=float(st.session_state.get("sef_builder_widener", 1.4)),
        aruco_quality_threshold=float(st.session_state.get("sef_builder_aruco_quality_threshold", 0.45)),
        aruco_alpha_high_quality=float(st.session_state.get("sef_builder_aruco_alpha_high_quality", 0.65)),
        aruco_alpha_low_quality=float(st.session_state.get("sef_builder_aruco_alpha_low_quality", 0.20)),
        aruco_max_jump_px=float(st.session_state.get("sef_builder_aruco_max_jump_px", 2.0)),
        aruco_smooth_corners=bool(st.session_state.get("sef_builder_aruco_smooth_corners", True)),
        smoothing_alpha=float(st.session_state.get("sef_builder_smoothing_alpha", 0.86)),
        smoothing_reset=float(st.session_state.get("sef_builder_smoothing_reset", 65.0)),
        background_method=str(st.session_state.get("sef_builder_bg_method", "MOG2")),
        background_shadows=bool(st.session_state.get("sef_builder_bg_shadows", False)),
        multi_max_objects=int(st.session_state.get("sef_builder_multi_max_objects", 4)),
        multi_similarity=float(st.session_state.get("sef_builder_multi_similarity", 0.62)),
        dense_cell_size=int(st.session_state.get("sef_builder_dense_cell_size", 16)),
        barrier_names=tuple(item.strip() for item in str(st.session_state.get("sef_builder_barrier_names", "")).split(",") if item.strip()),
        branching_rules=tuple(st.session_state.get("sef_builder_branching_rules", [])),
        color_stab_color_space=str(st.session_state.get("sef_builder_color_stab_color_space", "LAB")),
        color_stab_techniques=tuple(st.session_state.get("sef_builder_color_stab_techniques", ["luminance_normalization", "temporal_smoothing"])),
        color_stab_strength=float(st.session_state.get("sef_builder_color_stab_strength", 0.85)),
        color_stab_temporal_alpha=float(st.session_state.get("sef_builder_color_stab_temporal_alpha", 0.92)),
        color_stab_chroma=bool(st.session_state.get("sef_builder_color_stab_chroma", True)),
        color_stab_chroma_strength=float(st.session_state.get("sef_builder_color_stab_chroma_strength", 0.20)),
        color_stab_hist_min_std=float(st.session_state.get("sef_builder_color_stab_hist_min_std", 4.0)),
        color_stab_hist_max_gain=float(st.session_state.get("sef_builder_color_stab_hist_max_gain", 1.35)),
        color_stab_lum_max_shift=float(st.session_state.get("sef_builder_color_stab_lum_max_shift", 48.0)),
        color_stab_gamma=float(st.session_state.get("sef_builder_color_stab_gamma", 0.0)),
        color_stab_clahe_clip=float(st.session_state.get("sef_builder_color_stab_clahe_clip", 2.0)),
        color_stab_clahe_strength=float(st.session_state.get("sef_builder_color_stab_clahe_strength", 0.35)),
        color_stab_emit_metrics=bool(st.session_state.get("sef_builder_color_stab_emit_metrics", True)),
        color_stab_emit_overlay=bool(st.session_state.get("sef_builder_color_stab_emit_overlay", False)),
        color_stab_emit_intermediate=bool(st.session_state.get("sef_builder_color_stab_emit_intermediate", False)),
        dynamic_removal_sampling_stride=int(st.session_state.get("sef_builder_dynamic_removal_sampling_stride", 5)),
        dynamic_removal_max_sampled_frames=int(st.session_state.get("sef_builder_dynamic_removal_max_sampled_frames", 60)),
        dynamic_removal_difference_threshold=int(st.session_state.get("sef_builder_dynamic_removal_difference_threshold", 35)),
        dynamic_removal_morph_kernel_size=int(st.session_state.get("sef_builder_dynamic_removal_morph_kernel_size", 5)),
        dynamic_removal_opening_iterations=int(st.session_state.get("sef_builder_dynamic_removal_opening_iterations", 1)),
        dynamic_removal_closing_iterations=int(st.session_state.get("sef_builder_dynamic_removal_closing_iterations", 2)),
        dynamic_removal_dilation_iterations=int(st.session_state.get("sef_builder_dynamic_removal_dilation_iterations", 1)),
        dynamic_removal_min_component_area=int(st.session_state.get("sef_builder_dynamic_removal_min_component_area", 80)),
        dynamic_removal_max_processed_frames=int(st.session_state.get("sef_builder_dynamic_removal_max_processed_frames", 300)),
        dynamic_removal_emit_intermediate=bool(st.session_state.get("sef_builder_dynamic_removal_emit_intermediate", False)),
        intermediate_capture_enabled=bool(st.session_state.get("sef_builder_intermediate_capture_enabled", False)),
        intermediate_capture_max_frames=int(st.session_state.get("sef_builder_intermediate_capture_max_frames", 30)),
        runtime_frame_buffer_size=int(st.session_state.get("sef_builder_runtime_frame_buffer_size", 8)),
        runtime_signal_buffer_size=int(st.session_state.get("sef_builder_runtime_signal_buffer_size", 8)),
        runtime_data_buffer_size=int(st.session_state.get("sef_builder_runtime_data_buffer_size", 8)),
        runtime_latency_policy=str(st.session_state.get("sef_builder_runtime_latency_policy", "blocking")),
        runtime_adaptive_min_interval=int(st.session_state.get("sef_builder_runtime_adaptive_min_interval", 1)),
        runtime_adaptive_max_interval=int(st.session_state.get("sef_builder_runtime_adaptive_max_interval", 8)),
        runtime_adaptive_low_watermark=float(st.session_state.get("sef_builder_runtime_adaptive_low_watermark", 0.25)),
        runtime_adaptive_high_watermark=float(st.session_state.get("sef_builder_runtime_adaptive_high_watermark", 0.75)),
    )


def stage_labels() -> dict[str, str]:
    """Return UI labels for stage keys."""
    return {stage.value: label for stage, label in STAGE_LABELS.items()}


def ordered_stage_options(names: list[str], recommended: set[str]) -> list[str]:
    """Return recommended plugins first, then every other available plugin."""
    recommended_items = [name for name in names if name in recommended]
    remaining_items = [name for name in names if name not in recommended]
    return recommended_items + remaining_items


def plugin_names(registry, category: PluginCategory) -> list[str]:
    """Return all plugin names for a registry category."""
    return [plugin.name for plugin in sorted(registry.list(category), key=lambda item: item.name)]


def first_plugin_name(registry, category: PluginCategory) -> str:
    """Return the first available plugin name for a category."""
    names = plugin_names(registry, category)
    return names[0] if names else ""


def display_plugin(registry, category: PluginCategory):
    """Create a formatter for registry plugin labels."""

    def _format(name: str) -> str:
        try:
            plugin = registry.get(category, name)
            return f"{plugin.name} - {plugin.factory.__name__}"
        except Exception:
            return name

    return _format


def display_stage_plugin(registry, category: PluginCategory, recommended: set[str]):
    """Create a formatter that also highlights recommendation state."""
    base_formatter = display_plugin(registry, category)

    def _format(name: str) -> str:
        suffix = " [consigliato]" if name in recommended else " [compatibilita non verificata]"
        return f"{base_formatter(name)}{suffix}"

    return _format


def display_frame_processor_plugin(registry):
    """Create a formatter for single-frame and frame-buffer processors."""
    recommended = recommended_frame_processors()

    def _format(name: str) -> str:
        category = (
            PluginCategory.FRAME_BUFFER_PROCESSOR
            if name in FRAME_BUFFER_PROCESSORS
            else PluginCategory.SINGLE_FRAME_PROCESSOR
        )
        suffix = " [consigliato]" if name in recommended else " [compatibilita non verificata]"
        try:
            plugin = registry.get(category, name)
            return f"{plugin.name} - {plugin.factory.__name__}{suffix}"
        except Exception:
            return f"{name}{suffix}"

    return _format


def selected_signal_extractor() -> str:
    """Return the selected signal extractor."""
    return builder_state().signal_extractor


def selected_resize() -> tuple[int, int] | None:
    """Return the selected resize tuple, if any."""
    return builder_state().resize


def recommended_frame_extractors() -> set[str]:
    return {"opencv_buffered", "opencv_webcam"}


def recommended_frame_processors() -> set[str]:
    extractor = selected_signal_extractor()
    if extractor in {"opencv_tracker", "opencv_stream_tracker", "opencv_multi_tracker"}:
        return {"smoothing"}
    return set()


def recommended_signal_extractors() -> set[str]:
    mode = builder_state().mode
    if mode == "Multi-object barriers":
        return {"opencv_multi_tracker"}
    if mode == "Dense optical flow":
        return {"dense_optical_flow"}
    if mode == "ArUco wall micromovements":
        return {"aruco_marker"}
    if mode == "YOLO pose realtime":
        return {YOLO_POSE_SIGNAL_EXTRACTOR}
    return {"opencv_tracker"}


def recommended_signal_cleaners() -> set[str]:
    if selected_signal_extractor() == "opencv_tracker":
        return {"moving_average"}
    if selected_signal_extractor() == "opencv_stream_tracker":
        return {"moving_average_stream"}
    if selected_signal_extractor() == "aruco_marker":
        return {"aruco_temporal_stabilizer"}
    return set()


def recommended_analyzers_for_current_signal() -> set[str]:
    extractor = selected_signal_extractor()
    if extractor == "opencv_multi_tracker":
        return {"barrier_counting", "tracking_playback"}
    if extractor == "dense_optical_flow":
        return {"dense_vector_field"}
    if extractor == "aruco_marker":
        return set(ARUCO_ANALYZERS)
    if extractor == YOLO_POSE_SIGNAL_EXTRACTOR:
        return set(YOLO_POSE_ANALYZERS)
    return set(TRACKING_ANALYZERS)


def recommended_visualizers_for_current_state() -> set[str]:
    selected_analyzers = set(builder_state().analyzers)
    recommended: set[str] = set()
    if selected_analyzers & set(TRACKING_ANALYZERS):
        recommended.add("matplotlib_function")
    if "barrier_counting" in selected_analyzers:
        recommended.add("matplotlib_histogram")
    if "dense_vector_field" in selected_analyzers:
        recommended.add("matplotlib_vector_field")
    if "tracking_playback" in selected_analyzers:
        recommended.add("tracking_video")
    if selected_analyzers & set(ARUCO_ANALYZERS):
        recommended.add("aruco_motion_plot")
    if "aruco_displacement" in selected_analyzers:
        recommended.add("aruco_annotated_video")
    if selected_analyzers & set(YOLO_POSE_ANALYZERS):
        recommended.add(STREAMLIT_COCO_POSE_REALTIME_VISUALIZER)
    return recommended


def recommended_intermediate_visualizers_for_current_state() -> set[str]:
    """Return intermediate-frame visualizers that fit the current debug settings."""
    state = builder_state()
    if state.intermediate_capture_enabled or state.color_stab_emit_intermediate or state.dynamic_removal_emit_intermediate:
        return {"intermediate_frames_grid"}
    return set()


def suggested_visualizer_target_indices(
    visualizer_name: str,
    analyzer_names: tuple[str, ...] | list[str],
) -> tuple[int, ...] | None:
    """Return compatible analyzer result indices for a visualizer."""
    analyzers = tuple(analyzer_names)
    compatibility_map = {
        "matplotlib_function": set(TRACKING_ANALYZERS),
        "matplotlib_histogram": {"barrier_counting"},
        "matplotlib_vector_field": {"dense_vector_field"},
        "matplotlib_heatmap": {"dense_vector_field"},
        "tracking_video": {"tracking_playback"},
        "aruco_motion_plot": set(ARUCO_ANALYZERS),
        "aruco_annotated_video": {"aruco_displacement"},
        "opencv_coco_pose_realtime": {"coco_pose_stream"},
        "opencv_coco_tennis_pose_realtime": {"coco_pose_stream"},
        STREAMLIT_COCO_POSE_REALTIME_VISUALIZER: {"coco_pose_stream"},
    }
    compatible_analyzers = compatibility_map.get(visualizer_name)
    if compatible_analyzers is None:
        return None
    return tuple(index for index, analyzer_name in enumerate(analyzers) if analyzer_name in compatible_analyzers)


def analyzer_options_for_current_signal(registry) -> list[str]:
    """Return analyzer options ordered by recommendation relevance."""
    names = plugin_names(registry, PluginCategory.ANALYZER)
    return ordered_stage_options(names, recommended_analyzers_for_current_signal())


def single_frame_processor_options_for_current_signal(registry) -> list[str]:
    """Return frame processor options ordered by recommendation relevance."""
    names = [
        *plugin_names(registry, PluginCategory.SINGLE_FRAME_PROCESSOR),
        *plugin_names(registry, PluginCategory.FRAME_BUFFER_PROCESSOR),
    ]
    return ordered_stage_options(
        sorted(set(names)),
        recommended_frame_processors(),
    )


def analysis_visualizer_options_for_current_state(registry) -> list[str]:
    """Return non-intermediate visualizers ordered by recommendation relevance."""
    names = [
        name
        for name in plugin_names(registry, PluginCategory.VISUALIZER)
        if name not in INTERMEDIATE_FRAME_VISUALIZERS
    ]
    return ordered_stage_options(names, recommended_visualizers_for_current_state())


def intermediate_visualizer_options_for_current_state(registry) -> list[str]:
    """Return visualizers that consume IntermediateFrameArtifactCollection."""
    names = [
        name
        for name in plugin_names(registry, PluginCategory.VISUALIZER)
        if name in INTERMEDIATE_FRAME_VISUALIZERS
    ]
    return ordered_stage_options(names, recommended_intermediate_visualizers_for_current_state())


def ensure_stage_options(registry):
    """Normalise selected widget values against the live registry."""
    frame_extractor_options = ordered_stage_options(
        plugin_names(registry, PluginCategory.FRAME_EXTRACTOR),
        recommended_frame_extractors(),
    )
    if st.session_state["sef_builder_frame_extractor"] not in frame_extractor_options:
        st.session_state["sef_builder_frame_extractor"] = frame_extractor_options[0] if frame_extractor_options else ""

    signal_extractor_options = ordered_stage_options(
        plugin_names(registry, PluginCategory.SIGNAL_EXTRACTOR),
        recommended_signal_extractors(),
    )
    if st.session_state["sef_builder_signal_extractor"] not in signal_extractor_options:
        st.session_state["sef_builder_signal_extractor"] = signal_extractor_options[0] if signal_extractor_options else ""

    frame_processor_options = single_frame_processor_options_for_current_signal(registry)
    st.session_state["sef_builder_frame_processors"] = [
        name for name in st.session_state.get("sef_builder_frame_processors", []) if name in frame_processor_options
    ]

    signal_cleaner_options = ordered_stage_options(
        plugin_names(registry, PluginCategory.SIGNAL_CLEANER),
        recommended_signal_cleaners(),
    )
    st.session_state["sef_builder_signal_cleaners"] = [
        name for name in st.session_state.get("sef_builder_signal_cleaners", []) if name in signal_cleaner_options
    ]

    analyzer_options = analyzer_options_for_current_signal(registry)
    st.session_state["sef_builder_analyzers"] = [name for name in st.session_state.get("sef_builder_analyzers", []) if name in analyzer_options]

    visualizer_options = analysis_visualizer_options_for_current_state(registry)
    intermediate_visualizer_options = intermediate_visualizer_options_for_current_state(registry)

    selected_visualizers = list(st.session_state.get("sef_builder_visualizers", []))
    selected_intermediate_visualizers = list(st.session_state.get("sef_builder_intermediate_visualizers", []))
    migrated_intermediate_visualizers = [
        name
        for name in selected_visualizers
        if name in INTERMEDIATE_FRAME_VISUALIZERS and name not in selected_intermediate_visualizers
    ]
    st.session_state["sef_builder_visualizers"] = [
        name for name in selected_visualizers if name in visualizer_options
    ]
    st.session_state["sef_builder_intermediate_visualizers"] = [
        name
        for name in [*selected_intermediate_visualizers, *migrated_intermediate_visualizers]
        if name in intermediate_visualizer_options
    ]

    return (
        frame_extractor_options,
        signal_extractor_options,
        signal_cleaner_options,
        analyzer_options,
        visualizer_options,
        intermediate_visualizer_options,
    )


def parse_indices(raw: str) -> tuple[int, ...] | None:
    """Parse a comma-separated list of result indices."""
    if not raw:
        return None
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def build_pipeline_configuration_from_state() -> PipelineConfiguration:
    """Convert the UI state snapshot into a typed pipeline configuration."""
    state = builder_state()
    video_path = session.get(session.VIDEO_PATH)
    roi = session.get(session.ROI_BOX) or (0, 0, 0, 0)
    barriers = session.get(session.BARRIERS) or {}

    return PipelineConfiguration(
        frame_extractor=_build_frame_extractor_config(state, video_path),
        signal_extractor=_build_signal_extractor_config(state, roi, video_path),
        frame_processors=_build_single_frame_processor_configs(state),
        signal_cleaners=_build_signal_cleaner_configs(state),
        analyzers=_build_analyzer_configs(state, barriers),
        visualizers=_build_visualizer_configs(state),
        intermediate_frames=_build_intermediate_frame_configuration(state),
        runtime=_build_runtime_configuration(state),
    )


def generated_pipeline_config_dict() -> dict[str, Any]:
    """Return the current generated config as a plain dict."""
    return build_pipeline_configuration_from_state().to_dict()


def _build_frame_extractor_config(
    state: BuilderStateSnapshot,
    video_path: str | None,
) -> PluginConfig:
    extractor_config = {
        "resize": state.resize,
        "stride": state.stride,
        "max_frames": state.max_frames if state.max_frames_enabled else None,
    }
    if state.frame_extractor == "opencv_webcam":
        if state.webcam_mirror:
            extractor_config["mirror"] = True
        if state.webcam_width > 0:
            extractor_config["width"] = state.webcam_width
        if state.webcam_height > 0:
            extractor_config["height"] = state.webcam_height
        if state.webcam_fps > 0:
            extractor_config["fps"] = state.webcam_fps
        return PluginConfig(
            name=state.frame_extractor,
            params={
                "camera_index": state.webcam_index,
                "config": extractor_config,
            },
        )
    return PluginConfig(
        name=state.frame_extractor,
        params={
            "path": video_path or "",
            "config": extractor_config,
        },
    )


def current_pipeline_config_dict() -> dict[str, Any]:
    """Return the effective config used by execution."""
    if not has_manual_config_override():
        return generated_pipeline_config_dict()
    config = session.get(session.PIPELINE_CONFIG)
    if isinstance(config, dict):
        return config
    return generated_pipeline_config_dict()


def has_manual_config_override() -> bool:
    """Return True only when the user explicitly enabled the JSON override."""
    return bool(session.get(session.PIPELINE_CONFIG_OVERRIDE_ENABLED, False))


def validate_runtime_requirements(config: dict[str, Any]) -> list[str]:
    """Validate runtime requirements against the current UI state."""
    issues: list[str] = []
    pipeline = dict(config.get("pipeline", {}))
    frame_extractor = dict(pipeline.get("frame_extractor", {}))
    signal_extractor = dict(pipeline.get("signal_extractor", {}))
    signal_extractor_params = dict(signal_extractor.get("params", {}))

    if (
        str(frame_extractor.get("name", "")) != "opencv_webcam"
        and not session.get(session.VIDEO_PATH)
        and not dict(frame_extractor.get("params", {})).get("path")
    ):
        issues.append("Seleziona un video nella tab Composer.")

    extractor_name = str(signal_extractor.get("name", ""))
    selected_analyzers = {str(item.get("name", "")) for item in pipeline.get("analyzers", [])}
    selected_frame_processors = {str(item.get("name", "")) for item in pipeline.get("frame_processors", [])}

    if extractor_name in {"opencv_tracker", "opencv_stream_tracker", "opencv_multi_tracker"}:
        roi = signal_extractor_params.get("start_box") or session.get(session.ROI_BOX)
        if not roi or roi[2] <= 0 or roi[3] <= 0:
            issues.append("Disegna una ROI valida per il tracker.")

    if extractor_name == "aruco_marker":
        resize = dict(frame_extractor.get("params", {})).get("config", {}).get("resize")
        if isinstance(resize, (tuple, list)) and len(resize) == 2 and (int(resize[0]) < 640 or int(resize[1]) < 480):
            issues.append("Per ArUco evita downscale troppo aggressivi: sotto 640x480 la detection puo degradare.")
        stride = dict(frame_extractor.get("params", {})).get("config", {}).get("stride")
        if isinstance(stride, int) and stride > 1:
            issues.append("Per ArUco usa Stride=1: saltare frame riduce stabilita e osservabilita del moto.")

    if extractor_name == "opencv_multi_tracker" and "barrier_counting" in selected_analyzers:
        configured_barriers = {}
        for analyzer in pipeline.get("analyzers", []):
            if str(analyzer.get("name", "")) != "barrier_counting":
                continue
            configured_barriers = dict(dict(analyzer.get("params", {})).get("barriers", {}))
            if configured_barriers:
                break
        barrier_state = session.get(session.BARRIER_SELECTION_STATE)
        if configured_barriers:
            pass
        elif not isinstance(barrier_state, BarrierSelectionState) or not barrier_state.complete:
            issues.append("Disegna tutte le barriere richieste per il conteggio multi-oggetto.")

    if extractor_name in {"opencv_multi_tracker", "dense_optical_flow"} and "opencv_gray" in selected_frame_processors:
        issues.append("Rimuovi opencv_gray: questo extractor richiede frame BGR.")

    if extractor_name == YOLO_POSE_SIGNAL_EXTRACTOR:
        visualizers = {str(item.get("name", "")) for item in pipeline.get("visualizers", [])}
        cleaners = {str(item.get("name", "")) for item in pipeline.get("signal_cleaners", [])}
        model_name = str(signal_extractor_params.get("model_name", "yolo11s-pose.pt"))
        model_path = Path(__file__).resolve().parents[2] / "sef" / "builtin" / "YOLOPoseModels" / model_name
        if not model_path.exists():
            issues.append(f"Modello YOLO non trovato: {model_path}.")
        if "coco_pose_stream" not in selected_analyzers:
            issues.append("Per YOLO pose seleziona l'analyzer coco_pose_stream.")
        if visualizers & YOLO_POSE_VISUALIZERS:
            issues.append(
                "I visualizer live OpenCV non sono eseguibili dentro Streamlit: "
                "rimuovili dalla UI oppure esegui questa pipeline da script/VSCode."
            )
        if visualizers & YOLO_POSE_VISUALIZERS and "coco_skeleton_normalization" in cleaners:
            issues.append(
                "Rimuovi coco_skeleton_normalization quando usi la preview live: il visualizer richiede keypoint in coordinate frame."
            )
        if "streamlit_coco_pose_realtime" in visualizers:
            frame_extractor_name = str(frame_extractor.get("name", ""))
            frame_config = dict(dict(frame_extractor.get("params", {})).get("config", {}))
            max_frames = frame_config.get("max_frames")
            if frame_extractor_name == "opencv_buffered" and max_frames is None:
                issues.append(
                    "Per preview realtime da file imposta un limite Max frames: "
                    "con max_frames=null il video puo restare in esecuzione molto a lungo."
                )

    if not pipeline.get("analyzers"):
        issues.append("Seleziona almeno un analyzer.")

    return issues


def main_thread_visualizer_names(config: dict[str, Any], registry) -> tuple[str, ...]:
    """Return configured visualizers that declare a main-thread rendering requirement."""
    names: list[str] = []
    pipeline = dict(config.get("pipeline", {}))
    for item in pipeline.get("visualizers", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        if not name:
            continue
        try:
            factory = registry.get(PluginCategory.VISUALIZER, name).factory
        except Exception:
            continue
        if bool(getattr(factory, "requires_main_thread", False)):
            names.append(name)
    return tuple(names)


def sync_mode_with_components(mode: str) -> None:
    """Apply scenario defaults only when the scenario selector changes."""
    if st.session_state.get(BUILDER_LAST_SYNCED_MODE_KEY) == mode:
        return

    _disable_json_override()
    st.session_state[BUILDER_LAST_SYNCED_MODE_KEY] = mode
    if mode == "Single object tracking":
        apply_tracking_components()
    elif mode == "Multi-object barriers":
        apply_multi_object_components()
    elif mode == "Dense optical flow":
        apply_dense_flow_components()
    elif mode == "ArUco wall micromovements":
        apply_aruco_components()
    elif mode == "YOLO pose realtime":
        apply_yolo_pose_components()


def apply_tracking_preset() -> None:
    _disable_json_override()
    st.session_state["sef_builder_mode"] = "Single object tracking"
    st.session_state[BUILDER_LAST_SYNCED_MODE_KEY] = "Single object tracking"
    apply_tracking_components()


def apply_tracking_components() -> None:
    st.session_state["sef_builder_signal_extractor"] = "opencv_tracker"
    st.session_state["sef_builder_signal_cleaners"] = ["moving_average"]
    st.session_state["sef_builder_analyzers"] = ["vertical_position", "vertical_velocity"]
    st.session_state["sef_builder_frame_processors"] = ["smoothing"]


def apply_multi_object_preset() -> None:
    _disable_json_override()
    st.session_state["sef_builder_mode"] = "Multi-object barriers"
    st.session_state[BUILDER_LAST_SYNCED_MODE_KEY] = "Multi-object barriers"
    apply_multi_object_components()


def apply_multi_object_components() -> None:
    st.session_state["sef_builder_signal_extractor"] = "opencv_multi_tracker"
    st.session_state["sef_builder_signal_cleaners"] = []
    st.session_state["sef_builder_analyzers"] = ["barrier_counting"]
    st.session_state["sef_builder_frame_processors"] = ["smoothing"]


def apply_dense_flow_preset() -> None:
    _disable_json_override()
    st.session_state["sef_builder_mode"] = "Dense optical flow"
    st.session_state[BUILDER_LAST_SYNCED_MODE_KEY] = "Dense optical flow"
    apply_dense_flow_components()


def apply_dense_flow_components() -> None:
    st.session_state["sef_builder_signal_extractor"] = "dense_optical_flow"
    st.session_state["sef_builder_signal_cleaners"] = []
    st.session_state["sef_builder_analyzers"] = ["dense_vector_field"]
    st.session_state["sef_builder_frame_processors"] = []


def apply_aruco_preset() -> None:
    _disable_json_override()
    st.session_state["sef_builder_mode"] = "ArUco wall micromovements"
    st.session_state[BUILDER_LAST_SYNCED_MODE_KEY] = "ArUco wall micromovements"
    apply_aruco_components()


def apply_aruco_components() -> None:
    st.session_state["sef_builder_signal_extractor"] = "aruco_marker"
    st.session_state["sef_builder_signal_cleaners"] = ["aruco_temporal_stabilizer"]
    st.session_state["sef_builder_analyzers"] = ["aruco_displacement"]
    st.session_state["sef_builder_frame_processors"] = []
    st.session_state["sef_builder_visualizers"] = ["aruco_motion_plot", "aruco_annotated_video"]
    # ArUco runs can otherwise explode RAM by buffering full-resolution videos.
    st.session_state["sef_builder_resize"] = "640x480"
    st.session_state["sef_builder_stride"] = 1
    st.session_state["sef_builder_max_frames_enabled"] = True
    st.session_state["sef_builder_max_frames"] = 180


def apply_yolo_pose_preset() -> None:
    _disable_json_override()
    st.session_state["sef_builder_mode"] = "YOLO pose realtime"
    st.session_state[BUILDER_LAST_SYNCED_MODE_KEY] = "YOLO pose realtime"
    apply_yolo_pose_components()


def apply_yolo_pose_components() -> None:
    st.session_state["sef_builder_frame_extractor"] = "opencv_webcam"
    st.session_state["sef_builder_signal_extractor"] = YOLO_POSE_SIGNAL_EXTRACTOR
    st.session_state["sef_builder_signal_cleaners"] = []
    st.session_state["sef_builder_analyzers"] = ["coco_pose_stream"]
    st.session_state["sef_builder_visualizers"] = [STREAMLIT_COCO_POSE_REALTIME_VISUALIZER]
    st.session_state["sef_builder_frame_processors"] = []
    st.session_state["sef_builder_resize"] = "640x480"
    st.session_state["sef_builder_stride"] = 1
    st.session_state["sef_builder_max_frames_enabled"] = True
    st.session_state["sef_builder_max_frames"] = 300
    st.session_state["sef_builder_runtime_frame_buffer_size"] = 1
    st.session_state["sef_builder_runtime_signal_buffer_size"] = 1
    st.session_state["sef_builder_runtime_data_buffer_size"] = 1
    st.session_state["sef_builder_runtime_latency_policy"] = "drop_oldest"
    st.session_state["sef_builder_yolo_include_frame_image"] = True
    st.session_state["sef_builder_yolo_draw_source_frame"] = True


def _disable_json_override() -> None:
    """Make generated composer state the execution source after scenario changes."""
    session.put(session.PIPELINE_CONFIG_OVERRIDE_ENABLED, False)


def _build_single_frame_processor_configs(state: BuilderStateSnapshot) -> tuple[PluginConfig, ...]:
    configs: list[PluginConfig] = []
    for name in state.frame_processors:
        if name == "opencv_resize" and state.resize is not None:
            configs.append(PluginConfig(name=name, params={"size": state.resize}))
        elif name == "smoothing":
            configs.append(
                PluginConfig(
                    name=name,
                    params={
                        "alpha": state.smoothing_alpha,
                        "reset_threshold": state.smoothing_reset,
                    },
                )
            )
        elif name == "background_subtraction":
            configs.append(
                PluginConfig(
                    name=name,
                    params={
                        "method": state.background_method,
                        "detect_shadows": state.background_shadows,
                    },
                )
            )
        elif name == "color_stabilization":
            configs.append(
                PluginConfig(
                    name=name,
                    params={
                        "color_space": state.color_stab_color_space,
                        "techniques": list(state.color_stab_techniques),
                        "stabilization_strength": state.color_stab_strength,
                        "temporal_alpha": state.color_stab_temporal_alpha,
                        "stabilize_chroma": state.color_stab_chroma,
                        "chroma_strength": state.color_stab_chroma_strength,
                        "histogram_min_std": state.color_stab_hist_min_std,
                        "histogram_max_gain": state.color_stab_hist_max_gain,
                        "luminance_max_shift": state.color_stab_lum_max_shift,
                        "gamma": state.color_stab_gamma if state.color_stab_gamma > 0 else None,
                        "clahe_clip_limit": state.color_stab_clahe_clip,
                        "clahe_strength": state.color_stab_clahe_strength,
                        "emit_metrics": state.color_stab_emit_metrics,
                        "emit_comparison_overlay": state.color_stab_emit_overlay,
                        "emit_intermediate_artifacts": state.color_stab_emit_intermediate,
                    },
                )
            )
        elif name == "dynamic_object_removal":
            configs.append(
                PluginConfig(
                    name=name,
                    processor_type="frame_buffer",
                    params={
                        "sampling_stride": state.dynamic_removal_sampling_stride,
                        "max_sampled_frames": state.dynamic_removal_max_sampled_frames,
                        "difference_threshold": state.dynamic_removal_difference_threshold,
                        "morph_kernel_size": state.dynamic_removal_morph_kernel_size,
                        "opening_iterations": state.dynamic_removal_opening_iterations,
                        "closing_iterations": state.dynamic_removal_closing_iterations,
                        "dilation_iterations": state.dynamic_removal_dilation_iterations,
                        "min_component_area": state.dynamic_removal_min_component_area,
                        "max_processed_frames": state.dynamic_removal_max_processed_frames,
                        "emit_intermediate_artifacts": state.dynamic_removal_emit_intermediate,
                    },
                )
            )
        elif name == "motion_magnification":
            configs.append(
                PluginConfig(
                    name=name,
                    processor_type="frame_buffer",
                )
            )
        else:
            configs.append(PluginConfig(name=name))
    return tuple(configs)


def _build_signal_extractor_config(
    state: BuilderStateSnapshot,
    roi: tuple[int, int, int, int],
    video_path: str | None,
) -> PluginConfig:
    if state.signal_extractor == "dense_optical_flow":
        return PluginConfig(
            name=state.signal_extractor,
            params={"cell_size": state.dense_cell_size},
        )
    if state.signal_extractor == "aruco_marker":
        return PluginConfig(
            name=state.signal_extractor,
            params={
                "config": {
                    "white_border_padding_px": 32,
                }
            },
        )
    if state.signal_extractor == YOLO_POSE_SIGNAL_EXTRACTOR:
        return PluginConfig(
            name=state.signal_extractor,
            params={
                "model_name": state.yolo_model_name,
                "config": {
                    "include_frame_image": state.yolo_include_frame_image,
                    "show_graph": False,
                },
            },
        )

    params: dict[str, Any] = {
        "tracker_type": state.tracker,
        "start_box": roi,
        "config": {
            "show": state.show_windows if state.signal_extractor != "opencv_stream_tracker" else False,
            "source_path": video_path,
        },
    }
    if state.signal_extractor == "opencv_multi_tracker":
        params.update(
            {
                "max_objects": state.multi_max_objects,
                "template_match_threshold": state.multi_similarity,
            }
        )
    return PluginConfig(name=state.signal_extractor, params=params)


def _build_signal_cleaner_configs(state: BuilderStateSnapshot) -> tuple[PluginConfig, ...]:
    configs: list[PluginConfig] = []
    for name in state.signal_cleaners:
        if name in {"moving_average", "moving_average_stream"}:
            configs.append(PluginConfig(name=name, params={"window_size": state.moving_average_window}))
        elif name == "outlier_rejection":
            configs.append(
                PluginConfig(
                    name=name,
                    params={
                        "threshold": state.outlier_threshold,
                        "mode": state.outlier_mode,
                    },
                )
            )
        elif name == "signal_widener":
            configs.append(PluginConfig(name=name, params={"amplification": state.widener}))
        elif name == "aruco_temporal_stabilizer":
            configs.append(
                PluginConfig(
                    name=name,
                    params={
                        "quality_threshold": state.aruco_quality_threshold,
                        "alpha_high_quality": state.aruco_alpha_high_quality,
                        "alpha_low_quality": state.aruco_alpha_low_quality,
                        "max_jump_px": state.aruco_max_jump_px,
                        "smooth_corners": state.aruco_smooth_corners,
                    },
                )
            )
        elif name == "coco_skeleton_normalization":
            configs.append(
                PluginConfig(
                    name=name,
                    params={
                        "center_on_pelvis": state.coco_norm_center_on_pelvis,
                        "normalize_scale": state.coco_norm_normalize_scale,
                        "align_rotation": state.coco_norm_align_rotation,
                        "min_scale": state.coco_norm_min_scale,
                    },
                )
            )
        else:
            configs.append(PluginConfig(name=name))
    return tuple(configs)


def _build_analyzer_configs(
    state: BuilderStateSnapshot,
    barriers: dict[str, Any],
) -> tuple[PluginConfig, ...]:
    configs: list[PluginConfig] = []
    for name in state.analyzers:
        if name == "barrier_counting":
            configs.append(PluginConfig(name=name, params={"barriers": barriers}))
        elif name in TRACKING_ANALYZERS:
            configs.append(PluginConfig(name=name, params={"config": {"use_timestamps": True}}))
        elif name == "coco_pose_stream":
            configs.append(
                PluginConfig(
                    name=name,
                    params={
                        "config": {
                            "retain_frames": state.coco_pose_retain_frames,
                            "include_normalized_skeleton": state.coco_pose_include_normalized_skeleton,
                        }
                    },
                )
            )
        else:
            configs.append(PluginConfig(name=name))
    return tuple(configs)


def _build_visualizer_configs(state: BuilderStateSnapshot) -> tuple[VisualizerConfig, ...]:
    visualizers: list[VisualizerConfig] = []
    for name in state.visualizers:
        if name in INTERMEDIATE_FRAME_VISUALIZERS:
            continue
        raw_indices = str(state.visualizer_targets.get(name, "")).strip()
        indices = parse_indices(raw_indices) if raw_indices else suggested_visualizer_target_indices(name, state.analyzers)
        config: dict[str, Any] = {"show": False}
        if name in YOLO_POSE_VISUALIZERS or name in STREAMLIT_REALTIME_VISUALIZERS:
            config = {
                "draw_source_frame": state.yolo_draw_source_frame,
                "keypoint_threshold": state.yolo_keypoint_threshold,
                "min_publish_interval_ms": 40,
            }
        visualizers.append(
            VisualizerConfig(
                name=name,
                params={"config": config},
                result_indices=indices,
            )
        )
    return tuple(visualizers)


def _build_intermediate_frame_configuration(
    state: BuilderStateSnapshot,
) -> IntermediateFrameConfiguration | None:
    """Build the dedicated intermediate-frame config section, if needed."""
    processor_debug_capture_enabled = state.color_stab_emit_intermediate or state.color_stab_emit_overlay
    if not state.intermediate_capture_enabled and not state.intermediate_visualizers and not processor_debug_capture_enabled:
        return None
    return IntermediateFrameConfiguration(
        enabled=True,
        max_stored_frames=state.intermediate_capture_max_frames,
        visualizers=tuple(
            PluginConfig(name=name, params={"config": {"show": False}})
            for name in state.intermediate_visualizers
        ),
    )


def _build_runtime_configuration(state: BuilderStateSnapshot) -> RuntimeConfiguration:
    """Build the explicit pipeline.runtime section used by adaptive execution."""
    params: dict[str, Any] = {}
    if state.runtime_latency_policy == "adaptive_sampling":
        params = {
            "min_interval": state.runtime_adaptive_min_interval,
            "max_interval": state.runtime_adaptive_max_interval,
            "low_watermark": state.runtime_adaptive_low_watermark,
            "high_watermark": state.runtime_adaptive_high_watermark,
        }
    return RuntimeConfiguration(
        frame_buffer_size=state.runtime_frame_buffer_size,
        signal_buffer_size=state.runtime_signal_buffer_size,
        data_buffer_size=state.runtime_data_buffer_size,
        latency_policy_name=state.runtime_latency_policy,
        latency_policy_params=params,
    )
