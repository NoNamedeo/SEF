"""State and configuration services for the Streamlit composer."""

from __future__ import annotations

from typing import Any

import streamlit as st

from library.core.plugins.PluginRegistry import PluginCategory
from ui.models.pipeline_builder import (
    AnalysisStageKey,
    BarrierSelectionState,
    BuilderStateSnapshot,
    PipelineConfiguration,
    PluginConfig,
    STAGE_LABELS,
    VisualizerConfig,
)
from ui.state import session

TRACKING_ANALYZERS = (
    "vertical_position",
    "horizontal_position",
    "vertical_velocity",
    "horizontal_velocity",
    "vertical_frequency",
    "horizontal_frequency",
)

STAGE_EDIT_OPTIONS = tuple(stage.value for stage in AnalysisStageKey)


def initialise_builder_state(registry) -> None:
    """Populate the Streamlit session with explicit builder defaults."""
    selected_stage = st.query_params.get("stage", AnalysisStageKey.FRAME_EXTRACTOR.value)
    defaults = {
        "sef_builder_frame_extractor": first_plugin_name(registry, PluginCategory.FRAME_EXTRACTOR),
        "sef_builder_frame_cleaners": ["smoothing"],
        "sef_builder_signal_extractor": "opencv_tracker",
        "sef_builder_signal_cleaners": ["moving_average"],
        "sef_builder_analyzers": ["vertical_position"],
        "sef_builder_visualizers": [],
        "sef_builder_visualizer_targets": {},
        "sef_builder_mode": "Single object tracking",
        "sef_builder_resize": "640x480",
        "sef_builder_stride": 2,
        "sef_builder_max_frames_enabled": True,
        "sef_builder_max_frames": 180,
        "sef_builder_tracker": "MIL",
        "sef_builder_show_windows": False,
        "sef_builder_mavg_window": 5,
        "sef_builder_outlier_threshold": 3.5,
        "sef_builder_outlier_mode": "clip",
        "sef_builder_widener": 1.4,
        "sef_builder_smoothing_alpha": 0.86,
        "sef_builder_smoothing_reset": 65.0,
        "sef_builder_bg_method": "MOG2",
        "sef_builder_bg_shadows": False,
        "sef_builder_multi_max_objects": 4,
        "sef_builder_multi_similarity": 0.62,
        "sef_builder_dense_cell_size": 16,
        "sef_builder_barrier_names": "A, B",
        "sef_builder_branching_rules": [],
        "sef_selected_stage": selected_stage if selected_stage in STAGE_EDIT_OPTIONS else AnalysisStageKey.FRAME_EXTRACTOR.value,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def builder_state() -> BuilderStateSnapshot:
    """Return a typed snapshot of the current composer state."""
    return BuilderStateSnapshot(
        mode=str(st.session_state.get("sef_builder_mode", "Single object tracking")),
        frame_extractor=str(st.session_state.get("sef_builder_frame_extractor", "")),
        frame_cleaners=tuple(st.session_state.get("sef_builder_frame_cleaners", [])),
        signal_extractor=str(st.session_state.get("sef_builder_signal_extractor", "opencv_tracker")),
        signal_cleaners=tuple(st.session_state.get("sef_builder_signal_cleaners", [])),
        analyzers=tuple(st.session_state.get("sef_builder_analyzers", [])),
        visualizers=tuple(st.session_state.get("sef_builder_visualizers", [])),
        visualizer_targets=dict(st.session_state.get("sef_builder_visualizer_targets", {})),
        resize_label=str(st.session_state.get("sef_builder_resize", "640x480")),
        stride=int(st.session_state.get("sef_builder_stride", 2)),
        max_frames_enabled=bool(st.session_state.get("sef_builder_max_frames_enabled", True)),
        max_frames=int(st.session_state.get("sef_builder_max_frames", 180)),
        tracker=str(st.session_state.get("sef_builder_tracker", "MIL")),
        show_windows=bool(st.session_state.get("sef_builder_show_windows", False)),
        moving_average_window=int(st.session_state.get("sef_builder_mavg_window", 5)),
        outlier_threshold=float(st.session_state.get("sef_builder_outlier_threshold", 3.5)),
        outlier_mode=str(st.session_state.get("sef_builder_outlier_mode", "clip")),
        widener=float(st.session_state.get("sef_builder_widener", 1.4)),
        smoothing_alpha=float(st.session_state.get("sef_builder_smoothing_alpha", 0.86)),
        smoothing_reset=float(st.session_state.get("sef_builder_smoothing_reset", 65.0)),
        background_method=str(st.session_state.get("sef_builder_bg_method", "MOG2")),
        background_shadows=bool(st.session_state.get("sef_builder_bg_shadows", False)),
        multi_max_objects=int(st.session_state.get("sef_builder_multi_max_objects", 4)),
        multi_similarity=float(st.session_state.get("sef_builder_multi_similarity", 0.62)),
        dense_cell_size=int(st.session_state.get("sef_builder_dense_cell_size", 16)),
        barrier_names=tuple(
            item.strip() for item in str(st.session_state.get("sef_builder_barrier_names", "")).split(",") if item.strip()
        ),
        branching_rules=tuple(st.session_state.get("sef_builder_branching_rules", [])),
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


def selected_signal_extractor() -> str:
    """Return the selected signal extractor."""
    return builder_state().signal_extractor


def selected_resize() -> tuple[int, int] | None:
    """Return the selected resize tuple, if any."""
    return builder_state().resize


def recommended_frame_extractors() -> set[str]:
    return {"opencv_buffered"}


def recommended_frame_cleaners() -> set[str]:
    extractor = selected_signal_extractor()
    if extractor in {"opencv_tracker", "opencv_multi_tracker"}:
        return {"smoothing"}
    return set()


def recommended_signal_extractors() -> set[str]:
    mode = builder_state().mode
    if mode == "Multi-object barriers":
        return {"opencv_multi_tracker"}
    if mode == "Dense optical flow":
        return {"dense_optical_flow"}
    return {"opencv_tracker"}


def recommended_signal_cleaners() -> set[str]:
    if selected_signal_extractor() == "opencv_tracker":
        return {"moving_average"}
    return set()


def recommended_analyzers_for_current_signal() -> set[str]:
    extractor = selected_signal_extractor()
    if extractor == "opencv_multi_tracker":
        return {"barrier_counting", "tracking_playback"}
    if extractor == "dense_optical_flow":
        return {"dense_vector_field"}
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
    return recommended


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
    }
    compatible_analyzers = compatibility_map.get(visualizer_name)
    if compatible_analyzers is None:
        return None
    indices = tuple(index for index, analyzer_name in enumerate(analyzers) if analyzer_name in compatible_analyzers)
    return indices or None


def analyzer_options_for_current_signal(registry) -> list[str]:
    """Return analyzer options ordered by recommendation relevance."""
    names = plugin_names(registry, PluginCategory.ANALYZER)
    return ordered_stage_options(names, recommended_analyzers_for_current_signal())


def frame_cleaner_options_for_current_signal(registry) -> list[str]:
    """Return frame cleaner options ordered by recommendation relevance."""
    return ordered_stage_options(
        plugin_names(registry, PluginCategory.FRAME_CLEANER),
        recommended_frame_cleaners(),
    )


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

    signal_cleaner_options = ordered_stage_options(
        plugin_names(registry, PluginCategory.SIGNAL_CLEANER),
        recommended_signal_cleaners(),
    )
    st.session_state["sef_builder_signal_cleaners"] = [
        name for name in st.session_state.get("sef_builder_signal_cleaners", []) if name in signal_cleaner_options
    ]

    analyzer_options = analyzer_options_for_current_signal(registry)
    st.session_state["sef_builder_analyzers"] = [
        name for name in st.session_state.get("sef_builder_analyzers", []) if name in analyzer_options
    ]

    visualizer_options = ordered_stage_options(
        plugin_names(registry, PluginCategory.VISUALIZER),
        recommended_visualizers_for_current_state(),
    )
    st.session_state["sef_builder_visualizers"] = [
        name for name in st.session_state.get("sef_builder_visualizers", []) if name in visualizer_options
    ]

    return (
        frame_extractor_options,
        signal_extractor_options,
        signal_cleaner_options,
        analyzer_options,
        visualizer_options,
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
        frame_extractor=PluginConfig(
            name=state.frame_extractor,
            params={
                "path": video_path or "",
                "config": {
                    "resize": state.resize,
                    "stride": state.stride,
                    "max_frames": state.max_frames if state.max_frames_enabled else None,
                },
            },
        ),
        signal_extractor=_build_signal_extractor_config(state, roi, video_path),
        frame_cleaners=_build_frame_cleaner_configs(state),
        signal_cleaners=_build_signal_cleaner_configs(state),
        analyzers=_build_analyzer_configs(state, barriers),
        visualizers=_build_visualizer_configs(state),
    )


def generated_pipeline_config_dict() -> dict[str, Any]:
    """Return the current generated config as a plain dict."""
    return build_pipeline_configuration_from_state().to_dict()


def current_pipeline_config_dict() -> dict[str, Any]:
    """Return the effective config used by execution."""
    config = session.get(session.PIPELINE_CONFIG)
    if isinstance(config, dict):
        return config
    return generated_pipeline_config_dict()


def validate_runtime_requirements(config: dict[str, Any]) -> list[str]:
    """Validate runtime requirements against the current UI state."""
    issues: list[str] = []
    pipeline = dict(config.get("pipeline", {}))
    frame_extractor = dict(pipeline.get("frame_extractor", {}))
    signal_extractor = dict(pipeline.get("signal_extractor", {}))
    signal_extractor_params = dict(signal_extractor.get("params", {}))

    if not session.get(session.VIDEO_PATH) and not dict(frame_extractor.get("params", {})).get("path"):
        issues.append("Seleziona un video nella tab Composer.")

    extractor_name = str(signal_extractor.get("name", ""))
    selected_analyzers = {str(item.get("name", "")) for item in pipeline.get("analyzers", [])}
    selected_frame_cleaners = {str(item.get("name", "")) for item in pipeline.get("frame_cleaners", [])}

    if extractor_name in {"opencv_tracker", "opencv_multi_tracker"}:
        roi = signal_extractor_params.get("start_box") or session.get(session.ROI_BOX)
        if not roi or roi[2] <= 0 or roi[3] <= 0:
            issues.append("Disegna una ROI valida per il tracker.")

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

    if extractor_name in {"opencv_multi_tracker", "dense_optical_flow"} and "opencv_gray" in selected_frame_cleaners:
        issues.append("Rimuovi opencv_gray: questo extractor richiede frame BGR.")

    if not pipeline.get("analyzers"):
        issues.append("Seleziona almeno un analyzer.")

    return issues


def sync_mode_with_components(mode: str) -> None:
    """Apply scenario defaults when the extractor no longer matches the chosen mode."""
    current = selected_signal_extractor()
    if mode == "Single object tracking" and current != "opencv_tracker":
        apply_tracking_components()
    elif mode == "Multi-object barriers" and current != "opencv_multi_tracker":
        apply_multi_object_components()
    elif mode == "Dense optical flow" and current != "dense_optical_flow":
        apply_dense_flow_components()


def apply_tracking_preset() -> None:
    st.session_state["sef_builder_mode"] = "Single object tracking"
    apply_tracking_components()


def apply_tracking_components() -> None:
    st.session_state["sef_builder_signal_extractor"] = "opencv_tracker"
    st.session_state["sef_builder_signal_cleaners"] = ["moving_average"]
    st.session_state["sef_builder_analyzers"] = ["vertical_position", "vertical_velocity"]
    st.session_state["sef_builder_frame_cleaners"] = ["smoothing"]


def apply_multi_object_preset() -> None:
    st.session_state["sef_builder_mode"] = "Multi-object barriers"
    apply_multi_object_components()


def apply_multi_object_components() -> None:
    st.session_state["sef_builder_signal_extractor"] = "opencv_multi_tracker"
    st.session_state["sef_builder_signal_cleaners"] = []
    st.session_state["sef_builder_analyzers"] = ["barrier_counting"]
    st.session_state["sef_builder_frame_cleaners"] = ["smoothing"]


def apply_dense_flow_preset() -> None:
    st.session_state["sef_builder_mode"] = "Dense optical flow"
    apply_dense_flow_components()


def apply_dense_flow_components() -> None:
    st.session_state["sef_builder_signal_extractor"] = "dense_optical_flow"
    st.session_state["sef_builder_signal_cleaners"] = []
    st.session_state["sef_builder_analyzers"] = ["dense_vector_field"]
    st.session_state["sef_builder_frame_cleaners"] = []


def _build_frame_cleaner_configs(state: BuilderStateSnapshot) -> tuple[PluginConfig, ...]:
    configs: list[PluginConfig] = []
    for name in state.frame_cleaners:
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

    params: dict[str, Any] = {
        "tracker_type": state.tracker,
        "start_box": roi,
        "config": {
            "show": state.show_windows,
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
        if name == "moving_average":
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
        else:
            configs.append(PluginConfig(name=name))
    return tuple(configs)


def _build_visualizer_configs(state: BuilderStateSnapshot) -> tuple[VisualizerConfig, ...]:
    visualizers: list[VisualizerConfig] = []
    for name in state.visualizers:
        raw_indices = str(state.visualizer_targets.get(name, "")).strip()
        indices = parse_indices(raw_indices) if raw_indices else suggested_visualizer_target_indices(name, state.analyzers)
        visualizers.append(
            VisualizerConfig(
                name=name,
                params={"config": {"show": False}},
                result_indices=indices,
            )
        )
    return tuple(visualizers)
