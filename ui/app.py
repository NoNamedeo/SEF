"""
SEF Studio.

Run with:
    streamlit run ui/app.py

This is the application-facing cockpit for the pipeline core: component
discovery, visual pipeline composition, video/ROI setup, execution, monitor
snapshots, event stream and runtime plugin registration.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from library.core.events.PipelineLifecycleEvent import PipelineLifecycleEvent  # noqa: E402
from library.core.plugins.PluginRegistry import PluginCategory  # noqa: E402
from ui.components.barrier_selector import render_barrier_selector  # noqa: E402
from ui.components.pipeline_canvas import render_pipeline_canvas  # noqa: E402
from ui.components.pipeline_status_dashboard import (  # noqa: E402
    render_event_timeline,
    render_pipeline_status_dashboard,
)
from ui.components.pipeline_outputs_viewer import render_pipeline_outputs  # noqa: E402
from ui.components.roi_selector import render_roi_selector  # noqa: E402
from ui.components.video_selector import render_video_selector  # noqa: E402
from ui.services.pipeline_canvas_service import build_pipeline_canvas_model  # noqa: E402
from ui.services.pipeline_service import (  # noqa: E402
    active_ids,
    cancel_async,
    clear_event_records,
    configure_branching_rules,
    context_from_config,
    dispatch_trigger,
    event_integration_status,
    event_records,
    pipeline_outputs,
    run_sync,
    snapshots,
    submit_async,
)
from ui.services.registry_bootstrap import get_registry  # noqa: E402
from ui.state import session  # noqa: E402
from ui.state.canvas import (  # noqa: E402
    ensure_canvas_state,
    last_query_stage,
    selected_stage,
    set_last_query_stage,
    set_selected_stage,
    sync_layout_from_query,
)

st.set_page_config(
    page_title="SEF Studio",
    page_icon="SEF",
    layout="wide",
    initial_sidebar_state="expanded",
)


STAGE_LABELS = {
    "frame_extractor": "Frame extractor",
    "frame_cleaners": "Frame cleaners",
    "signal_extractor": "Signal extractor",
    "signal_cleaners": "Signal cleaners",
    "analyzers": "Analyzers",
    "visualizers": "Visualizers",
}
STAGE_EDIT_OPTIONS = list(STAGE_LABELS.keys())

TRACKING_ANALYZERS = [
    "vertical_position",
    "horizontal_position",
    "vertical_velocity",
    "horizontal_velocity",
    "vertical_frequency",
    "horizontal_frequency",
]


def main() -> None:
    registry = get_registry()
    initialise_builder_state(registry)
    sync_stage_from_query()

    render_sidebar(registry)
    render_header(registry)

    tab_compose, tab_execute, tab_registry, tab_config = st.tabs(["Composer", "Run & Monitor", "Registry", "Config"])

    with tab_compose:
        render_composer(registry)

    with tab_execute:
        render_execution(registry)

    with tab_registry:
        render_registry(registry)

    with tab_config:
        render_config_lab(registry)


def initialise_builder_state(registry) -> None:
    ensure_canvas_state()
    selected_stage = st.query_params.get("stage", "frame_extractor")
    defaults = {
        "sef_builder_frame_extractor": first_plugin_name(registry, PluginCategory.FRAME_EXTRACTOR),
        "sef_builder_frame_cleaners": ["smoothing"],
        "sef_builder_signal_extractor": "opencv_tracker",
        "sef_builder_signal_cleaners": ["moving_average"],
        "sef_builder_analyzers": ["vertical_position"],
        "sef_builder_visualizers": [],
        "sef_builder_result_indices": "0",
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
        "sef_selected_stage": selected_stage if selected_stage in STAGE_EDIT_OPTIONS else "frame_extractor",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sync_stage_from_query() -> None:
    sync_layout_from_query()
    stage = st.query_params.get("stage")
    if stage in STAGE_EDIT_OPTIONS and stage != last_query_stage():
        set_selected_stage(stage)
        set_last_query_stage(stage)


def render_sidebar(registry) -> None:
    with st.sidebar:
        st.markdown("### SEF Studio")
        st.caption("Componi, esegui e osserva pipeline video dal core stabile.")

        st.divider()
        st.markdown("**Preset composizione**")
        if st.button("Tracking singolo", width="stretch"):
            apply_tracking_preset()
            st.rerun()
        if st.button("Multi-oggetto + barriere", width="stretch"):
            apply_multi_object_preset()
            st.rerun()
        if st.button("Dense optical flow", width="stretch"):
            apply_dense_flow_preset()
            st.rerun()

        st.divider()
        counts = {cat: len(registry.list(cat)) for cat in PluginCategory}
        st.markdown("**Componenti disponibili**")
        for cat in (
            PluginCategory.FRAME_EXTRACTOR,
            PluginCategory.FRAME_CLEANER,
            PluginCategory.SIGNAL_EXTRACTOR,
            PluginCategory.SIGNAL_CLEANER,
            PluginCategory.ANALYZER,
            PluginCategory.VISUALIZER,
            PluginCategory.BRANCHING_RULE,
        ):
            st.metric(cat.value.replace("_", " ").title(), counts.get(cat, 0))


def render_header(registry) -> None:
    total_plugins = len(registry.list())
    st.title("SEF Studio")
    st.caption(
        "Composer applicativo per pipeline video: scegli i componenti dal registry, "
        "configura ROI e barriere, esegui in sync o async e osserva eventi e stati."
    )
    cols = st.columns(5)
    cols[0].metric("Plugin registrati", total_plugins)
    cols[1].metric("Orchestrator", "attivo")
    cols[2].metric("EventBus", "opzionale")
    cols[3].metric("Monitor", "snapshot")
    cols[4].metric("Builder", "config")
    st.divider()


def render_composer(registry) -> None:
    st.subheader("Composizione visuale")
    mode = st.radio(
        "Scenario",
        ["Single object tracking", "Multi-object barriers", "Dense optical flow"],
        key="sef_builder_mode",
        horizontal=True,
    )
    sync_mode_with_components(mode)
    render_interactive_pipeline_board(registry)

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        render_video_and_geometry()
    with right:
        render_stage_parameter_editor(registry)
        st.divider()
        render_event_integration_editor(registry)

    st.divider()
    st.subheader("Configurazione generata")
    st.caption("Questa configurazione è quella che viene passata al ConfigPipelineBuilder.")
    st.json(build_pipeline_config_from_state(), expanded=False)


def render_video_and_geometry() -> None:
    st.markdown("### Sorgente e geometria")
    video_path, first_frame, meta = render_video_selector()

    if video_path:
        previous_video = session.get(session.VIDEO_PATH)
        if previous_video != video_path:
            for key in (session.ROI_BOX, session.BARRIERS, session.PIPELINE_OUTPUTS):
                session.clear(key)
        session.put(session.VIDEO_PATH, video_path)
        session.put(session.FIRST_FRAME, first_frame)
        session.put(session.FRAME_META, meta)
    else:
        for key in (
            session.VIDEO_PATH,
            session.FIRST_FRAME,
            session.FRAME_META,
            session.ROI_BOX,
            session.BARRIERS,
        ):
            session.clear(key)
        st.info("Seleziona un video per abilitare ROI, barriere ed esecuzione.")
        return

    frame = session.get(session.FIRST_FRAME)
    resize = selected_resize()
    last_resize_key = "studio_last_resize"
    previous_resize = st.session_state.get(last_resize_key, resize)
    if previous_resize != resize:
        for key in (session.ROI_BOX, session.BARRIERS, session.PIPELINE_OUTPUTS):
            session.clear(key)
    st.session_state[last_resize_key] = resize

    barrier_names = tuple(item.strip() for item in st.session_state.get("sef_builder_barrier_names", "").split(",") if item.strip())
    last_barrier_names_key = "studio_last_barrier_names"
    previous_barrier_names = st.session_state.get(last_barrier_names_key, barrier_names)
    if previous_barrier_names != barrier_names:
        for key in (session.BARRIERS, session.PIPELINE_OUTPUTS):
            session.clear(key)
    st.session_state[last_barrier_names_key] = barrier_names

    st.markdown("#### ROI di tracking")
    roi = render_roi_selector(frame, resize=resize, key="studio_roi")
    if roi:
        session.put(session.ROI_BOX, roi)
        st.success(f"ROI selezionata: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    if selected_signal_extractor() == "opencv_multi_tracker":
        st.markdown("#### Barriere")
        names = list(barrier_names)
        barriers = render_barrier_selector(
            frame,
            barrier_names=names,
            resize=resize,
            state_key="studio_barriers",
        )
        if barriers:
            session.put(session.BARRIERS, barriers)


def render_interactive_pipeline_board(registry) -> None:
    ensure_stage_options(registry)
    config = build_pipeline_config_from_state()
    model = build_pipeline_canvas_model(
        config=config,
        registry=registry,
        selected_stage=selected_stage(),
        runtime_issues=validate_runtime_requirements(config),
        run_snapshots=snapshots(),
        recent_events=event_records(),
    )
    render_pipeline_canvas(model)
    st.caption(
        "Canvas interattivo del core: trascina nodi, usa wheel per zoom, trascina lo sfondo per pan, "
        "apri i dettagli del nodo per vedere porte, eventi e parametri correnti."
    )


def render_stage_parameter_editor(registry) -> None:
    (
        frame_extractor_options,
        signal_extractor_options,
        compatible_signal_cleaners,
        analyzer_options,
        visualizer_options,
    ) = ensure_stage_options(registry)

    stage = st.segmented_control(
        "Editor stage",
        options=STAGE_EDIT_OPTIONS,
        format_func=lambda key: STAGE_LABELS[key],
        selection_mode="single",
        default=selected_stage(),
        key="sef_selected_stage",
    )
    if stage is None:
        stage = "frame_extractor"
    st.query_params["stage"] = stage
    set_last_query_stage(stage)

    st.markdown(f"### {STAGE_LABELS[stage]}")

    if stage == "frame_extractor":
        st.selectbox(
            "Frame extractor",
            frame_extractor_options,
            key="sef_builder_frame_extractor",
            format_func=display_plugin(registry, PluginCategory.FRAME_EXTRACTOR),
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
        return

    if stage == "frame_cleaners":
        st.multiselect(
            "Frame cleaners",
            plugin_names(registry, PluginCategory.FRAME_CLEANER),
            key="sef_builder_frame_cleaners",
            format_func=display_plugin(registry, PluginCategory.FRAME_CLEANER),
        )
        render_frame_cleaner_params()
        return

    if stage == "signal_extractor":
        st.selectbox(
            "Signal extractor",
            signal_extractor_options,
            key="sef_builder_signal_extractor",
            format_func=display_plugin(registry, PluginCategory.SIGNAL_EXTRACTOR),
        )
        render_signal_extractor_params()
        return

    if stage == "signal_cleaners":
        st.multiselect(
            "Signal cleaners",
            compatible_signal_cleaners,
            key="sef_builder_signal_cleaners",
            format_func=display_plugin(registry, PluginCategory.SIGNAL_CLEANER),
            disabled=not compatible_signal_cleaners,
            help="Non disponibili per multi-object e dense optical flow.",
        )
        render_signal_cleaner_params()
        return

    if stage == "analyzers":
        st.multiselect(
            "Analyzers",
            analyzer_options,
            key="sef_builder_analyzers",
            format_func=display_plugin(registry, PluginCategory.ANALYZER),
        )
        return

    if stage == "visualizers":
        st.multiselect(
            "Pipeline visualizers opzionali",
            visualizer_options,
            key="sef_builder_visualizers",
            format_func=display_plugin(registry, PluginCategory.VISUALIZER),
            help="La UI renderizza già i risultati. Qui abiliti anche i visualizer del core.",
        )
        if st.session_state["sef_builder_visualizers"]:
            st.text_input(
                "Result indices",
                key="sef_builder_result_indices",
                help="Esempio: 0,1. Vuoto = tutti i risultati.",
            )


def render_event_integration_editor(registry) -> None:
    st.markdown("### Event & Branching")
    status = event_integration_status()
    branching_options = plugin_names(registry, PluginCategory.BRANCHING_RULE)

    c1, c2, c3 = st.columns(3)
    c1.metric("Lifecycle bus", "enabled" if status["lifecycle_bus"] else "idle")
    c2.metric("Domain bus", "enabled" if status["domain_bus"] else "idle")
    c3.metric("Branching rules", len(status["branching_rules"]))

    with st.expander("Lifecycle & domain channels", expanded=False):
        st.caption("Il core già espone lifecycle events del runner e domain events dei componenti `IEventEmitter`.")
        st.markdown("**Lifecycle events**")
        st.code("\n".join(str(item) for item in PipelineLifecycleEvent), language="text")
        domain_types = sorted({event.event_type for event in event_records() if not event.event_type.startswith("pipeline.")})
        st.markdown("**Domain events osservati**")
        if domain_types:
            st.code("\n".join(domain_types), language="text")
        else:
            st.info("Nessun domain event osservato in questa sessione.")

    with st.expander("Trigger event-driven", expanded=True):
        st.caption("Invia il context corrente al path `PipelineEvent -> PipelineOrchestrator -> runner`.")
        trigger_pipeline_id = st.text_input(
            "Trigger pipeline id",
            value=f"trigger-{uuid.uuid4().hex[:8]}",
            key="sef_trigger_pipeline_id",
        )
        if st.button("Dispatch trigger event", width="stretch"):
            try:
                context = context_from_config(build_pipeline_config_from_state(), registry)
                dispatch_trigger(trigger_pipeline_id, context)
                st.success(f"Trigger dispatchato per `{trigger_pipeline_id}`.")
                time.sleep(0.2)
                st.rerun()
            except Exception as exc:
                st.error(f"Trigger fallito: {exc}")

    with st.expander("Branching rules", expanded=False):
        st.caption(
            "Se nel registry sono presenti plugin `branching_rule`, puoi attivare il coordinatore "
            "che ascolta i domain events e genera pipeline secondarie."
        )
        if not branching_options:
            st.info("Nessuna branching rule registrata. Registrane una dalla tab Registry.")
        selected_rules = st.multiselect(
            "Branching rules attive",
            branching_options,
            key="sef_builder_branching_rules",
            format_func=display_plugin(registry, PluginCategory.BRANCHING_RULE),
            disabled=not branching_options,
        )
        if st.button("Apply branching wiring", width="stretch", disabled=not branching_options):
            try:
                ok, message = configure_branching_rules(registry, selected_rules)
                if ok:
                    st.success(message)
                else:
                    st.warning(message)
            except Exception as exc:
                st.error(f"Configurazione branching fallita: {exc}")


def ensure_stage_options(registry):
    frame_extractor_options = plugin_names(registry, PluginCategory.FRAME_EXTRACTOR)
    if st.session_state["sef_builder_frame_extractor"] not in frame_extractor_options:
        st.session_state["sef_builder_frame_extractor"] = frame_extractor_options[0] if frame_extractor_options else ""

    signal_extractor_options = plugin_names(registry, PluginCategory.SIGNAL_EXTRACTOR)
    if st.session_state["sef_builder_signal_extractor"] not in signal_extractor_options:
        st.session_state["sef_builder_signal_extractor"] = signal_extractor_options[0] if signal_extractor_options else ""

    compatible_signal_cleaners = (
        []
        if selected_signal_extractor() in {"opencv_multi_tracker", "dense_optical_flow"}
        else plugin_names(registry, PluginCategory.SIGNAL_CLEANER)
    )
    st.session_state["sef_builder_signal_cleaners"] = [
        name for name in st.session_state.get("sef_builder_signal_cleaners", []) if name in compatible_signal_cleaners
    ]

    analyzer_options = analyzer_options_for_current_signal(registry)
    st.session_state["sef_builder_analyzers"] = [name for name in st.session_state.get("sef_builder_analyzers", []) if name in analyzer_options]

    visualizer_options = plugin_names(registry, PluginCategory.VISUALIZER)
    return (
        frame_extractor_options,
        signal_extractor_options,
        compatible_signal_cleaners,
        analyzer_options,
        visualizer_options,
    )


def render_component_picker(registry) -> None:
    st.markdown("### Componenti pipeline")

    mode = st.radio(
        "Scenario",
        ["Single object tracking", "Multi-object barriers", "Dense optical flow"],
        key="sef_builder_mode",
        horizontal=True,
    )
    sync_mode_with_components(mode)

    frame_extractor_options = plugin_names(registry, PluginCategory.FRAME_EXTRACTOR)
    if st.session_state["sef_builder_frame_extractor"] not in frame_extractor_options:
        st.session_state["sef_builder_frame_extractor"] = frame_extractor_options[0] if frame_extractor_options else ""
    st.selectbox(
        "Frame extractor",
        frame_extractor_options,
        key="sef_builder_frame_extractor",
        format_func=display_plugin(registry, PluginCategory.FRAME_EXTRACTOR),
    )

    with st.expander("Frame extraction params", expanded=True):
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

    st.multiselect(
        "Frame cleaners",
        plugin_names(registry, PluginCategory.FRAME_CLEANER),
        key="sef_builder_frame_cleaners",
        format_func=display_plugin(registry, PluginCategory.FRAME_CLEANER),
    )
    render_frame_cleaner_params()

    signal_extractor_options = plugin_names(registry, PluginCategory.SIGNAL_EXTRACTOR)
    if st.session_state["sef_builder_signal_extractor"] not in signal_extractor_options:
        st.session_state["sef_builder_signal_extractor"] = signal_extractor_options[0] if signal_extractor_options else ""
    st.selectbox(
        "Signal extractor",
        signal_extractor_options,
        key="sef_builder_signal_extractor",
        format_func=display_plugin(registry, PluginCategory.SIGNAL_EXTRACTOR),
    )
    render_signal_extractor_params()

    compatible_signal_cleaners = (
        []
        if selected_signal_extractor() in {"opencv_multi_tracker", "dense_optical_flow"}
        else plugin_names(registry, PluginCategory.SIGNAL_CLEANER)
    )
    st.session_state["sef_builder_signal_cleaners"] = [
        name for name in st.session_state.get("sef_builder_signal_cleaners", []) if name in compatible_signal_cleaners
    ]
    st.multiselect(
        "Signal cleaners",
        compatible_signal_cleaners,
        key="sef_builder_signal_cleaners",
        format_func=display_plugin(registry, PluginCategory.SIGNAL_CLEANER),
        disabled=not compatible_signal_cleaners,
    )
    render_signal_cleaner_params()

    analyzer_options = analyzer_options_for_current_signal(registry)
    st.session_state["sef_builder_analyzers"] = [name for name in st.session_state.get("sef_builder_analyzers", []) if name in analyzer_options]
    st.multiselect(
        "Analyzers",
        analyzer_options,
        key="sef_builder_analyzers",
        format_func=display_plugin(registry, PluginCategory.ANALYZER),
    )

    visualizer_options = plugin_names(registry, PluginCategory.VISUALIZER)
    st.multiselect(
        "Pipeline visualizers opzionali",
        visualizer_options,
        key="sef_builder_visualizers",
        format_func=display_plugin(registry, PluginCategory.VISUALIZER),
        help="La UI renderizza già i risultati. Qui abiliti anche i visualizer del core.",
    )
    if st.session_state["sef_builder_visualizers"]:
        st.text_input(
            "Result indices per visualizer",
            key="sef_builder_result_indices",
            help="Esempio: 0,1. Vuoto = ogni visualizer riceve tutti i risultati.",
        )


def render_frame_cleaner_params() -> None:
    selected = set(st.session_state["sef_builder_frame_cleaners"])
    if not selected:
        return
    with st.expander("Frame cleaner params", expanded=False):
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


def render_execution(registry) -> None:
    st.subheader("Run & Monitor")
    config = build_pipeline_config_from_state()
    issues = validate_runtime_requirements(config)

    status_cols = st.columns(4)
    status_cols[0].metric("Active", len(active_ids()))
    status_cols[1].metric("Snapshots", len(snapshots()))
    status_cols[2].metric("Events", len(event_records()))
    status_cols[3].metric("Analyzers", len(config["pipeline"]["analyzers"]))

    if issues:
        for issue in issues:
            st.warning(issue)

    col_run, col_monitor = st.columns([0.55, 1.45], gap="large")
    with col_run:
        st.markdown("### Esecuzione")
        pipeline_id = st.text_input(
            "Pipeline ID",
            value=f"ui-{uuid.uuid4().hex[:8]}",
            key="sef_run_pipeline_id",
        )
        sync_clicked = st.button("Run sync", type="primary", width="stretch", disabled=bool(issues))
        async_clicked = st.button("Submit async", width="stretch", disabled=bool(issues))

        if sync_clicked:
            execute_sync(registry, config, pipeline_id)
        if async_clicked:
            execute_async(registry, config, pipeline_id)

        outputs = session.get(session.PIPELINE_OUTPUTS)
        if outputs:
            render_pipeline_outputs(outputs, title="Current run outputs")

    with col_monitor:
        render_pipeline_status_dashboard(snapshots(), event_records(), title="Pipeline status")
        render_event_timeline(event_records())
        render_stored_outputs_browser()
        with st.expander("Controls", expanded=False):
            c1, c2 = st.columns(2)
            if c1.button("Refresh", width="stretch"):
                st.rerun()
            if c2.button("Clear events", width="stretch"):
                clear_event_records()
                st.rerun()

            active = active_ids()
            if active:
                st.markdown("**Cancel best-effort**")
                for pipeline_id in active:
                    if st.button(f"Cancel {pipeline_id}", key=f"cancel_{pipeline_id}"):
                        cancelled = cancel_async(pipeline_id)
                        st.toast("Cancelled queued pipeline." if cancelled else "Pipeline already running or unknown.")
                        st.rerun()


def render_registry(registry) -> None:
    st.subheader("Registry operativo")
    st.caption("Componenti disponibili e registrazione runtime di nuovi plugin.")

    categories = [
        PluginCategory.FRAME_EXTRACTOR,
        PluginCategory.FRAME_CLEANER,
        PluginCategory.SIGNAL_EXTRACTOR,
        PluginCategory.SIGNAL_CLEANER,
        PluginCategory.ANALYZER,
        PluginCategory.VISUALIZER,
        PluginCategory.BRANCHING_RULE,
    ]

    selected_category = st.selectbox(
        "Categoria",
        categories,
        format_func=lambda cat: cat.value.replace("_", " ").title(),
    )
    rows = [
        {
            "name": plugin.name,
            "factory": plugin.factory.__name__,
            "description": plugin.description,
        }
        for plugin in sorted(registry.list(selected_category), key=lambda item: item.name)
    ]
    st.dataframe(rows, hide_index=True, width="stretch")

    st.markdown("### Registra nuovo componente")
    with st.form("register_plugin_form"):
        category = st.selectbox(
            "Categoria plugin",
            categories,
            format_func=lambda cat: cat.value.replace("_", " ").title(),
        )
        name = st.text_input("Nome registry", placeholder="my_custom_analyzer")
        class_path = st.text_input("Classe Python", placeholder="my_package.module.MyAnalyzer")
        description = st.text_input("Descrizione")
        submitted = st.form_submit_button("Registra componente")

    if submitted:
        register_runtime_plugin(registry, category, name, class_path, description)


def render_config_lab(registry) -> None:
    st.subheader("Config lab")
    current_config = build_pipeline_config_from_state()
    raw_config = st.text_area(
        "Config JSON",
        value=json.dumps(current_config, indent=2),
        height=440,
    )

    c1, c2 = st.columns(2)
    if c1.button("Validate config", width="stretch"):
        try:
            context = context_from_config(json.loads(raw_config), registry)
            st.success(
                "Config valida: "
                f"{len(context.frame_cleaners)} frame cleaner, "
                f"{len(context.signal_cleaners)} signal cleaner, "
                f"{len(context.analyzers)} analyzer."
            )
        except Exception as exc:
            st.error(f"Config non valida: {exc}")

    if c2.button("Run config", type="primary", width="stretch"):
        try:
            config = json.loads(raw_config)
            context = context_from_config(config, registry)
            outputs = run_sync(context, pipeline_id=f"config-{uuid.uuid4().hex[:8]}")
            session.put(session.PIPELINE_OUTPUTS, outputs)
            st.success(f"Pipeline completata: {len(outputs.results)} risultati, {len(outputs.artifacts)} artifact.")
            render_pipeline_outputs(outputs, title="Config outputs")
        except Exception as exc:
            st.error(f"Esecuzione fallita: {exc}")


def execute_sync(registry, config: dict[str, Any], pipeline_id: str) -> None:
    try:
        context = context_from_config(config, registry)
        with st.spinner("Pipeline in esecuzione..."):
            outputs = run_sync(context, pipeline_id=pipeline_id)
        session.put(session.PIPELINE_OUTPUTS, outputs)
        st.success(f"Pipeline completata: {len(outputs.results)} risultati, {len(outputs.artifacts)} artifact.")
    except Exception as exc:
        st.error(f"Pipeline fallita: {exc}")


def execute_async(registry, config: dict[str, Any], pipeline_id: str) -> None:
    try:
        context = context_from_config(config, registry)
        submitted_id = submit_async(pipeline_id, context)
        st.success(f"Pipeline {submitted_id} sottomessa in background.")
        time.sleep(0.2)
        st.rerun()
    except Exception as exc:
        st.error(f"Submit fallita: {exc}")


def build_pipeline_config_from_state() -> dict[str, Any]:
    video_path = session.get(session.VIDEO_PATH)
    resize = selected_resize()
    extractor = selected_signal_extractor()

    pipeline = {
        "frame_extractor": {
            "name": st.session_state["sef_builder_frame_extractor"],
            "params": {
                "path": video_path or "",
                "config": {
                    "resize": resize,
                    "stride": st.session_state["sef_builder_stride"],
                    "max_frames": (st.session_state["sef_builder_max_frames"] if st.session_state["sef_builder_max_frames_enabled"] else None),
                },
            },
        },
        "frame_cleaners": build_frame_cleaner_config(),
        "signal_extractor": build_signal_extractor_config(extractor),
        "signal_cleaners": build_signal_cleaner_config(),
        "analyzers": build_analyzer_config(extractor),
        "visualizers": build_visualizer_config(),
    }
    return {"pipeline": pipeline}


def build_frame_cleaner_config() -> list[dict[str, Any]]:
    configs = []
    for name in st.session_state["sef_builder_frame_cleaners"]:
        if name == "opencv_resize":
            resize = selected_resize()
            if resize is not None:
                configs.append({"name": name, "params": {"size": resize}})
        elif name == "smoothing":
            configs.append(
                {
                    "name": name,
                    "params": {
                        "alpha": st.session_state["sef_builder_smoothing_alpha"],
                        "reset_threshold": st.session_state["sef_builder_smoothing_reset"],
                    },
                }
            )
        elif name == "background_subtraction":
            configs.append(
                {
                    "name": name,
                    "params": {
                        "method": st.session_state["sef_builder_bg_method"],
                        "detect_shadows": st.session_state["sef_builder_bg_shadows"],
                    },
                }
            )
        else:
            configs.append({"name": name})
    return configs


def build_signal_extractor_config(extractor: str) -> dict[str, Any]:
    if extractor == "dense_optical_flow":
        return {
            "name": extractor,
            "params": {"cell_size": st.session_state["sef_builder_dense_cell_size"]},
        }

    roi = session.get(session.ROI_BOX) or (0, 0, 0, 0)
    params: dict[str, Any] = {
        "tracker_type": st.session_state["sef_builder_tracker"],
        "start_box": roi,
        "config": {
            "show": st.session_state["sef_builder_show_windows"],
            "source_path": session.get(session.VIDEO_PATH),
        },
    }
    if extractor == "opencv_multi_tracker":
        params.update(
            {
                "max_objects": st.session_state["sef_builder_multi_max_objects"],
                "similarity_threshold": st.session_state["sef_builder_multi_similarity"],
            }
        )
    return {"name": extractor, "params": params}


def render_stored_outputs_browser() -> None:
    available_ids = [snapshot.pipeline_id for snapshot in snapshots() if pipeline_outputs(snapshot.pipeline_id) is not None]
    if not available_ids:
        return

    st.markdown("### Stored outputs")
    selected_pipeline_id = st.selectbox(
        "Inspect pipeline outputs",
        available_ids,
        index=len(available_ids) - 1,
        key="sef_stored_output_pipeline_id",
    )
    outputs = pipeline_outputs(selected_pipeline_id)
    if outputs is not None:
        render_pipeline_outputs(outputs, title=selected_pipeline_id)


def build_signal_cleaner_config() -> list[dict[str, Any]]:
    if selected_signal_extractor() in {"opencv_multi_tracker", "dense_optical_flow"}:
        return []

    configs = []
    for name in st.session_state["sef_builder_signal_cleaners"]:
        if name == "moving_average":
            configs.append({"name": name, "params": {"window_size": st.session_state["sef_builder_mavg_window"]}})
        elif name == "outlier_rejection":
            configs.append(
                {
                    "name": name,
                    "params": {
                        "threshold": st.session_state["sef_builder_outlier_threshold"],
                        "mode": st.session_state["sef_builder_outlier_mode"],
                    },
                }
            )
        elif name == "signal_widener":
            configs.append({"name": name, "params": {"amplification": st.session_state["sef_builder_widener"]}})
        else:
            configs.append({"name": name})
    return configs


def build_analyzer_config(extractor: str) -> list[dict[str, Any]]:
    configs = []
    barriers = session.get(session.BARRIERS) or {}
    for name in st.session_state["sef_builder_analyzers"]:
        if name == "barrier_counting":
            configs.append({"name": name, "params": {"barriers": barriers}})
        elif name in TRACKING_ANALYZERS:
            configs.append({"name": name, "params": {"config": {"use_timestamps": True}}})
        else:
            configs.append({"name": name})

    if extractor == "opencv_multi_tracker" and not configs:
        configs.append({"name": "barrier_counting", "params": {"barriers": barriers}})
    if extractor == "dense_optical_flow" and not configs:
        configs.append({"name": "dense_vector_field"})
    return configs


def build_visualizer_config() -> list[dict[str, Any]]:
    raw_indices = st.session_state.get("sef_builder_result_indices", "").strip()
    try:
        indices = parse_indices(raw_indices)
    except ValueError:
        st.warning("Result indices non validi: usa numeri separati da virgola, esempio `0,1`.")
        indices = None
    visualizers = []
    for name in st.session_state["sef_builder_visualizers"]:
        entry: dict[str, Any] = {"name": name, "params": {"config": {"show": False}}}
        if indices is not None:
            entry["result_indices"] = indices
        visualizers.append(entry)
    return visualizers


def render_pipeline_board(config: dict[str, Any]) -> None:
    pipeline = config["pipeline"]
    cards = {
        "frame_extractor": [pipeline["frame_extractor"]["name"]],
        "frame_cleaners": [item["name"] for item in pipeline["frame_cleaners"]],
        "signal_extractor": [pipeline["signal_extractor"]["name"]],
        "signal_cleaners": [item["name"] for item in pipeline["signal_cleaners"]],
        "analyzers": [item["name"] for item in pipeline["analyzers"]],
        "visualizers": [item["name"] for item in pipeline["visualizers"]],
    }
    columns = st.columns(len(STAGE_LABELS))
    for column, (key, label) in zip(columns, STAGE_LABELS.items()):
        with column:
            with st.container(border=True):
                st.caption(label)
                items = cards[key]
                if items:
                    for item in items:
                        st.write(f"- `{item}`")
                else:
                    st.write("_nessun componente_")


def validate_runtime_requirements(config: dict[str, Any]) -> list[str]:
    issues = []
    if not session.get(session.VIDEO_PATH):
        issues.append("Seleziona un video nella tab Composer.")
    extractor = config["pipeline"]["signal_extractor"]["name"]
    if extractor in {"opencv_tracker", "opencv_multi_tracker"}:
        roi = session.get(session.ROI_BOX)
        if not roi or roi[2] <= 0 or roi[3] <= 0:
            issues.append("Disegna una ROI valida per il tracker.")
    if extractor == "opencv_multi_tracker" and not session.get(session.BARRIERS):
        issues.append("Disegna almeno una barriera per il conteggio multi-oggetto.")
    if extractor in {"opencv_multi_tracker", "dense_optical_flow"} and "opencv_gray" in st.session_state["sef_builder_frame_cleaners"]:
        issues.append("Rimuovi opencv_gray: questo extractor richiede frame BGR.")
    if not config["pipeline"]["analyzers"]:
        issues.append("Seleziona almeno un analyzer.")
    return issues


def register_runtime_plugin(registry, category: PluginCategory, name: str, class_path: str, description: str) -> None:
    if not name.strip():
        st.error("Il nome registry è obbligatorio.")
        return
    if "." not in class_path:
        st.error("Inserisci un path completo del tipo package.module.ClassName.")
        return
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        factory = getattr(module, class_name)
        registry.register(category, name.strip(), factory, description.strip())
        st.success(f"Plugin {name} registrato in {category.value}.")
        st.rerun()
    except Exception as exc:
        st.error(f"Registrazione fallita: {exc}")


def sync_mode_with_components(mode: str) -> None:
    current = selected_signal_extractor()
    if mode == "Single object tracking" and current not in {"opencv_tracker"}:
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


def analyzer_options_for_current_signal(registry) -> list[str]:
    names = plugin_names(registry, PluginCategory.ANALYZER)
    extractor = selected_signal_extractor()
    if extractor == "opencv_multi_tracker":
        return [name for name in names if name == "barrier_counting"]
    if extractor == "dense_optical_flow":
        return [name for name in names if name == "dense_vector_field"]
    return [name for name in names if name in TRACKING_ANALYZERS]


def selected_signal_extractor() -> str:
    return st.session_state.get("sef_builder_signal_extractor", "opencv_tracker")


def selected_resize() -> tuple[int, int] | None:
    value = st.session_state.get("sef_builder_resize", "640x480")
    if value == "Originale":
        return None
    width, height = value.split("x")
    return int(width), int(height)


def plugin_names(registry, category: PluginCategory) -> list[str]:
    return [plugin.name for plugin in sorted(registry.list(category), key=lambda item: item.name)]


def first_plugin_name(registry, category: PluginCategory) -> str:
    names = plugin_names(registry, category)
    return names[0] if names else ""


def display_plugin(registry, category: PluginCategory):
    def _format(name: str) -> str:
        try:
            plugin = registry.get(category, name)
            return f"{plugin.name} - {plugin.factory.__name__}"
        except Exception:
            return name

    return _format


def parse_indices(raw: str) -> list[int] | None:
    if not raw:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def fmt_time(value: float | None) -> str:
    if value is None:
        return "-"
    return time.strftime("%H:%M:%S", time.localtime(value))


if __name__ == "__main__":
    main()
