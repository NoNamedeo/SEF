"""Presentation components for pipeline stage editing."""

from __future__ import annotations

import streamlit as st

from library.core.plugins.PluginRegistry import PluginCategory
from ui.services.pipeline_builder_service import (
    STAGE_EDIT_OPTIONS,
    display_stage_plugin,
    ensure_stage_options,
    recommended_analyzers_for_current_signal,
    recommended_frame_extractors,
    recommended_frame_processors,
    recommended_intermediate_visualizers_for_current_state,
    recommended_signal_cleaners,
    recommended_signal_extractors,
    recommended_visualizers_for_current_state,
    selected_signal_extractor,
    single_frame_processor_options_for_current_signal,
    stage_labels,
    suggested_visualizer_target_indices,
)
from ui.state.canvas import last_query_stage, selected_stage, set_last_query_stage


def render_stage_parameter_editor(registry) -> None:
    """Render the stage-focused editor panel."""
    (
        frame_extractor_options,
        signal_extractor_options,
        signal_cleaner_options,
        analyzer_options,
        visualizer_options,
        intermediate_visualizer_options,
    ) = ensure_stage_options(registry)
    labels = stage_labels()

    stage = st.segmented_control(
        "Editor stage",
        options=STAGE_EDIT_OPTIONS,
        format_func=lambda key: labels[key],
        selection_mode="single",
        default=selected_stage(),
        key="sef_selected_stage",
    )
    if stage is None:
        stage = STAGE_EDIT_OPTIONS[0]
    if stage != last_query_stage():
        st.query_params["stage"] = stage
    set_last_query_stage(stage)

    st.markdown(f"### {labels[stage]}")

    if stage == "frame_extractor":
        _render_frame_extractor_editor(registry, frame_extractor_options)
        return
    if stage == "frame_processors":
        _render_frame_processors_editor(registry)
        return
    if stage == "signal_extractor":
        _render_signal_extractor_editor(registry, signal_extractor_options)
        return
    if stage == "signal_cleaners":
        _render_signal_cleaners_editor(registry, signal_cleaner_options)
        return
    if stage == "analyzers":
        _render_analyzers_editor(registry, analyzer_options)
        return
    if stage == "visualizers":
        _render_visualizers_editor(registry, visualizer_options, intermediate_visualizer_options)


def _render_frame_extractor_editor(registry, frame_extractor_options: list[str]) -> None:
    st.selectbox(
        "Frame extractor",
        frame_extractor_options,
        key="sef_builder_frame_extractor",
        format_func=display_stage_plugin(registry, PluginCategory.FRAME_EXTRACTOR, recommended_frame_extractors()),
    )
    c1, c2, c3 = st.columns(3)
    c1.selectbox(
        "Resize",
        ["Originale", "320x180", "640x360", "640x480", "960x540"],
        key="sef_builder_resize",
    )
    c2.slider("Stride", 1, 12, key="sef_builder_stride")
    c3.checkbox("Limita frame", key="sef_builder_max_frames_enabled")
    if st.session_state["sef_builder_max_frames_enabled"]:
        st.slider("Max frame", 20, 1200, key="sef_builder_max_frames", step=20)

    # Intermediate frame capture controls
    st.divider()
    st.caption("Intermediate frame capture for debugging & comparison")
    render_intermediate_frame_capture_params()


def _render_frame_processors_editor(registry) -> None:
    st.multiselect(
        "Frame processors",
        single_frame_processor_options_for_current_signal(registry),
        key="sef_builder_frame_processors",
        format_func=display_stage_plugin(registry, PluginCategory.SINGLE_FRAME_PROCESSOR, recommended_frame_processors()),
        help="I consigliati compaiono in alto. Le altre opzioni restano selezionabili senza garanzia di compatibilita.",
    )
    render_single_frame_processor_params()


def _render_signal_extractor_editor(registry, signal_extractor_options: list[str]) -> None:
    st.selectbox(
        "Signal extractor",
        signal_extractor_options,
        key="sef_builder_signal_extractor",
        format_func=display_stage_plugin(registry, PluginCategory.SIGNAL_EXTRACTOR, recommended_signal_extractors()),
    )
    render_signal_extractor_params()


def _render_signal_cleaners_editor(registry, signal_cleaner_options: list[str]) -> None:
    st.multiselect(
        "Signal cleaners",
        signal_cleaner_options,
        key="sef_builder_signal_cleaners",
        format_func=display_stage_plugin(registry, PluginCategory.SIGNAL_CLEANER, recommended_signal_cleaners()),
        help="I consigliati compaiono in alto. Le altre opzioni restano selezionabili senza garanzia di compatibilita.",
    )
    render_signal_cleaner_params()


def _render_analyzers_editor(registry, analyzer_options: list[str]) -> None:
    st.multiselect(
        "Analyzers",
        analyzer_options,
        key="sef_builder_analyzers",
        format_func=display_stage_plugin(registry, PluginCategory.ANALYZER, recommended_analyzers_for_current_signal()),
        help="I consigliati compaiono in alto. Le altre opzioni restano selezionabili senza garanzia di compatibilita.",
    )


def _render_visualizers_editor(
    registry,
    visualizer_options: list[str],
    intermediate_visualizer_options: list[str],
) -> None:
    st.multiselect(
        "Pipeline visualizers opzionali",
        visualizer_options,
        key="sef_builder_visualizers",
        format_func=display_stage_plugin(registry, PluginCategory.VISUALIZER, recommended_visualizers_for_current_state()),
        help="La UI rende sempre i dati analitici. Qui abiliti i visualizer del core e i loro artifact.",
    )
    if st.session_state["sef_builder_visualizers"]:
        render_visualizer_target_inputs()

    st.divider()
    st.multiselect(
        "Intermediate frame visualizers",
        intermediate_visualizer_options,
        key="sef_builder_intermediate_visualizers",
        format_func=display_stage_plugin(
            registry,
            PluginCategory.VISUALIZER,
            recommended_intermediate_visualizers_for_current_state(),
        ),
        help=(
            "Questi visualizer consumano solo IntermediateFrameArtifactCollection e vengono "
            "generati nella sezione pipeline.intermediate_frames."
        ),
    )
    if st.session_state.get("sef_builder_intermediate_visualizers"):
        st.session_state["sef_builder_intermediate_capture_enabled"] = True


def render_single_frame_processor_params() -> None:
    selected = set(st.session_state["sef_builder_frame_processors"])
    if not selected:
        return
    with st.expander("Frame processor params", expanded=False):
        if "opencv_resize" in selected:
            st.caption("opencv_resize usa lo stesso resize del frame extractor.")
        if "smoothing" in selected:
            c1, c2 = st.columns(2)
            c1.slider("Smoothing alpha", 0.1, 1.0, key="sef_builder_smoothing_alpha", step=0.02)
            c2.slider("Reset threshold", 5.0, 150.0, key="sef_builder_smoothing_reset", step=5.0)
        if "background_subtraction" in selected:
            c1, c2 = st.columns(2)
            c1.selectbox("Background method", ["MOG2", "KNN"], key="sef_builder_bg_method")
            c2.checkbox("Detect shadows", key="sef_builder_bg_shadows")
        if "color_stabilization" in selected:
            render_color_stabilization_params()


def render_signal_extractor_params() -> None:
    extractor = selected_signal_extractor()
    with st.expander("Signal extractor params", expanded=True):
        if extractor in {"opencv_tracker", "opencv_multi_tracker"}:
            c1, c2 = st.columns(2)
            c1.selectbox("Tracker", ["MIL", "KCF", "CSRT"], key="sef_builder_tracker")
            c2.checkbox("OpenCV preview windows", key="sef_builder_show_windows")
            if extractor == "opencv_multi_tracker":
                st.text_input("Nomi barriere", key="sef_builder_barrier_names")
                c3, c4 = st.columns(2)
                c3.slider("Max objects", 1, 12, key="sef_builder_multi_max_objects")
                c4.slider("Similarity", 0.20, 0.99, key="sef_builder_multi_similarity", step=0.01)
        elif extractor == "dense_optical_flow":
            st.slider("Cell size", 6, 48, key="sef_builder_dense_cell_size")
        elif extractor == "aruco_marker":
            st.caption("Per ArUco e consigliato usare Resize=Originale e Stride=1 nel frame extractor.")


def render_signal_cleaner_params() -> None:
    selected = set(st.session_state.get("sef_builder_signal_cleaners", []))
    if not selected:
        return
    with st.expander("Signal cleaner params", expanded=False):
        if "moving_average" in selected:
            st.slider("Moving average window", 3, 31, key="sef_builder_mavg_window", step=2)
        if "outlier_rejection" in selected:
            c1, c2 = st.columns(2)
            c1.slider("Outlier threshold", 1.0, 8.0, key="sef_builder_outlier_threshold", step=0.1)
            c2.selectbox("Outlier mode", ["clip", "replace", "remove"], key="sef_builder_outlier_mode")
        if "signal_widener" in selected:
            st.slider("Amplification", 0.5, 4.0, key="sef_builder_widener", step=0.1)
        if "aruco_temporal_stabilizer" in selected:
            c1, c2 = st.columns(2)
            c1.slider("Aruco quality threshold", 0.0, 1.0, key="sef_builder_aruco_quality_threshold", step=0.01)
            c2.slider("Aruco max jump (px)", 0.0, 10.0, key="sef_builder_aruco_max_jump_px", step=0.1)
            c3, c4 = st.columns(2)
            c3.slider("Aruco alpha high quality", 0.0, 1.0, key="sef_builder_aruco_alpha_high_quality", step=0.01)
            c4.slider("Aruco alpha low quality", 0.0, 1.0, key="sef_builder_aruco_alpha_low_quality", step=0.01)
            st.checkbox("Smooth ArUco corners", key="sef_builder_aruco_smooth_corners")


def render_visualizer_target_inputs() -> None:
    target_map = dict(st.session_state.get("sef_builder_visualizer_targets", {}))
    active_visualizers = list(st.session_state.get("sef_builder_visualizers", []))
    stale_names = [name for name in target_map if name not in active_visualizers]
    for name in stale_names:
        target_map.pop(name, None)
    st.session_state["sef_builder_visualizer_targets"] = target_map

    analyzer_names = tuple(st.session_state.get("sef_builder_analyzers", []))
    st.caption("Configura i result indices per singolo visualizer. Vuoto = target compatibili automatici.")
    for name in active_visualizers:
        field_key = f"sef_builder_visualizer_target__{name}"
        if field_key not in st.session_state:
            st.session_state[field_key] = target_map.get(name, "")
        suggested_indices = suggested_visualizer_target_indices(name, analyzer_names)
        suggested_text = ",".join(str(index) for index in suggested_indices) if suggested_indices is not None else ""
        value = st.text_input(
            f"Result indices per `{name}`",
            key=field_key,
            help="Esempio: 0,1. Se vuoto, la UI usa gli output compatibili con quel visualizer.",
            placeholder=suggested_text or "default automatico",
        )
        target_map[name] = value.strip()
    st.session_state["sef_builder_visualizer_targets"] = target_map


def render_color_stabilization_params() -> None:
    """Render parameter controls for the ColorStabilizationFrameProcessor."""
    with st.expander("Color stabilization params", expanded=True):
        st.caption(
            "Stabilizza luminosita, illuminazione e cromia tra frame consecutivi. "
            "Applicabile a qualsiasi scenario per ridurre flickering e derive cromatiche."
        )
        c1, c2 = st.columns(2)
        c1.selectbox(
            "Color space",
            ["RGB", "HSV", "LAB", "YCrCb"],
            key="sef_builder_color_stab_color_space",
        )
        c2.multiselect(
            "Techniques",
            [
                "luminance_normalization",
                "temporal_smoothing",
                "histogram_normalization",
                "clahe",
                "gamma_correction",
            ],
            key="sef_builder_color_stab_techniques",
        )
        c1, c2 = st.columns(2)
        c1.slider(
            "Stabilization strength",
            0.0,
            1.0,
            key="sef_builder_color_stab_strength",
            step=0.05,
        )
        c2.slider(
            "Temporal alpha",
            0.0,
            1.0,
            key="sef_builder_color_stab_temporal_alpha",
            step=0.01,
        )

        st.divider()
        st.checkbox("Stabilize chroma", key="sef_builder_color_stab_chroma")
        col_c1, col_c2 = st.columns(2)
        col_c1.slider(
            "Chroma strength",
            0.0,
            1.0,
            key="sef_builder_color_stab_chroma_strength",
            step=0.05,
        )
        col_c2.slider(
            "Histogram min std",
            0.5,
            16.0,
            key="sef_builder_color_stab_hist_min_std",
            step=0.5,
        )

        col_h1, col_h2 = st.columns(2)
        col_h1.slider(
            "Histogram max gain",
            1.0,
            2.0,
            key="sef_builder_color_stab_hist_max_gain",
            step=0.05,
        )
        col_h2.slider(
            "Luminance max shift",
            5.0,
            100.0,
            key="sef_builder_color_stab_lum_max_shift",
            step=1.0,
        )

        col_g1, col_g2 = st.columns(2)
        col_g1.slider(
            "Manual gamma (0=auto)",
            0.0,
            2.5,
            key="sef_builder_color_stab_gamma",
            step=0.05,
        )
        col_g2.slider(
            "CLAHE clip limit",
            0.5,
            8.0,
            key="sef_builder_color_stab_clahe_clip",
            step=0.5,
        )

        st.slider(
            "CLAHE strength",
            0.0,
            1.0,
            key="sef_builder_color_stab_clahe_strength",
            step=0.05,
        )

        st.divider()
        st.caption("Artifact emission settings")
        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.checkbox("Emit metrics", key="sef_builder_color_stab_emit_metrics")
        col_e2.checkbox(
            "Emit comparison overlay",
            key="sef_builder_color_stab_emit_overlay",
            help="Genera una side-by-side original vs processed per ogni frame.",
        )
        col_e3.checkbox(
            "Emit intermediate artifacts",
            key="sef_builder_color_stab_emit_intermediate",
            help="Registra i frame intermedi di lavoro e luminanza per ispezione.",
        )


def render_intermediate_frame_capture_params() -> None:
    """Render controls for enabling intermediate frame capture in the pipeline."""
    st.checkbox(
        "Enable intermediate frame capture",
        key="sef_builder_intermediate_capture_enabled",
        help="Cattura i frame intermedi di ogni single-frame processor per confronto e debug.",
    )
    if st.session_state.get("sef_builder_intermediate_capture_enabled", False):
        st.slider(
            "Max captured frames",
            5,
            120,
            key="sef_builder_intermediate_capture_max_frames",
            step=5,
            help="Numero massimo di frame intermedi da mantenere in memoria.",
        )
