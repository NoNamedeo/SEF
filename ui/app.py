"""
SEF Studio.

Run with:
    streamlit run ui/app.py
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
from ui.components.composer_geometry import render_video_and_geometry  # noqa: E402
from ui.components.composer_stage_editor import render_stage_parameter_editor  # noqa: E402
from ui.components.execution_plan_viewer import render_execution_plan  # noqa: E402
from ui.components.pipeline_canvas import render_pipeline_canvas  # noqa: E402
from ui.components.pipeline_outputs_viewer import render_pipeline_outputs  # noqa: E402
from ui.components.pipeline_status_dashboard import (  # noqa: E402
    render_event_timeline,
    render_pipeline_status_dashboard,
)
from ui.components.realtime_preview_viewer import render_realtime_preview  # noqa: E402
from ui.services.execution_plan_service import build_execution_plan_preview, summarize_execution_plan  # noqa: E402
from ui.services.pipeline_builder_service import (  # noqa: E402
    apply_aruco_preset,
    apply_dense_flow_preset,
    apply_multi_object_preset,
    apply_tracking_preset,
    apply_yolo_pose_preset,
    current_pipeline_config_dict,
    generated_pipeline_config_dict,
    has_manual_config_override,
    initialise_builder_state,
    main_thread_visualizer_names,
    stage_labels,
    sync_mode_with_components,
    validate_runtime_requirements,
)
from ui.services.pipeline_canvas_service import build_pipeline_canvas_model  # noqa: E402
from ui.services.pipeline_config_editor import (  # noqa: E402
    config_to_text,
    parse_config_text,
    should_refresh_editor_widget,
    sync_editor_text,
)
from ui.services.pipeline_service import (  # noqa: E402
    active_ids,
    cancel_async,
    clear_event_records,
    configure_branching_rules,
    configure_runner,
    context_from_config,
    dispatch_trigger,
    event_integration_status,
    event_records,
    pipeline_outputs,
    run_sync,
    runner_parallelism,
    snapshots,
    submit_async,
)
from ui.services.realtime_preview_service import (  # noqa: E402
    config_has_streamlit_realtime_visualizer,
    preview_has_content,
    reset_sink,
    streamlit_realtime_visualizer_names,
    with_realtime_sink_ids,
)
from ui.services.registry_bootstrap import get_registry  # noqa: E402
from ui.services.ui_log_service import (  # noqa: E402
    available_log_levels,
    clear_log_records,
    install_ui_log_capture,
    log_records,
    set_capture_level,
)
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


def main() -> None:
    install_ui_log_capture()
    registry = get_registry()
    ensure_canvas_state()
    initialise_builder_state(registry)
    sync_stage_from_query()

    render_sidebar(registry)
    render_header(registry)

    tab_compose, tab_plan, tab_execute, tab_registry, tab_config = st.tabs(["Composer", "Plan & Logs", "Run & Monitor", "Registry", "Config"])

    with tab_compose:
        render_composer(registry)

    with tab_plan:
        render_plan_and_logs(registry)

    with tab_execute:
        render_execution(registry)

    with tab_registry:
        render_registry(registry)

    with tab_config:
        render_config_lab(registry)


def sync_stage_from_query() -> None:
    sync_layout_from_query()
    stage = st.query_params.get("stage")
    labels = stage_labels()
    if stage in labels and stage != last_query_stage():
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
        if st.button("ArUco wall motion", width="stretch"):
            apply_aruco_preset()
            st.rerun()
        if st.button("YOLO pose realtime", width="stretch"):
            apply_yolo_pose_preset()
            st.rerun()

        st.divider()
        counts = {category: len(registry.list(category)) for category in PluginCategory}
        st.markdown("**Componenti disponibili**")
        for category in (
            PluginCategory.FRAME_EXTRACTOR,
            PluginCategory.SINGLE_FRAME_PROCESSOR,
            PluginCategory.FRAME_BUFFER_PROCESSOR,
            PluginCategory.SIGNAL_EXTRACTOR,
            PluginCategory.SIGNAL_CLEANER,
            PluginCategory.ANALYZER,
            PluginCategory.VISUALIZER,
            PluginCategory.BRANCHING_RULE,
        ):
            st.metric(category.value.replace("_", " ").title(), counts.get(category, 0))


def render_header(registry) -> None:
    st.title("SEF Studio")
    st.caption(
        "Composer applicativo per pipeline video: scegli componenti dal registry, "
        "controlla runtime stream/batch, osserva piano, eventi, output e log."
    )


def render_composer(registry) -> None:
    st.subheader("Composizione visuale")
    mode = st.radio(
        "Scenario",
        [
            "Single object tracking",
            "Multi-object barriers",
            "Dense optical flow",
            "ArUco wall micromovements",
            "YOLO pose realtime",
        ],
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
    render_composer_config_editor()


def render_plan_and_logs(registry) -> None:
    st.subheader("Plan & Logs")
    config = current_pipeline_config_dict()
    preview = build_execution_plan_preview(config, registry)
    if preview.error:
        st.error(f"Planner non disponibile: {preview.error}")
    elif preview.plan is not None:
        render_execution_plan(preview.plan, title="Piano runtime sincronizzato")
    else:
        st.info("Nessun piano disponibile per la configurazione corrente.")

    st.divider()
    render_log_terminal(key_prefix="plan")


def render_interactive_pipeline_board(registry) -> None:
    generated_config = generated_pipeline_config_dict()
    model = build_pipeline_canvas_model(
        config=generated_config,
        registry=registry,
        selected_stage=selected_stage(),
        runtime_issues=validate_runtime_requirements(generated_config),
        run_snapshots=snapshots(),
        recent_events=event_records(),
    )
    render_pipeline_canvas(model)
    st.caption(
        "Canvas interattivo del core: trascina nodi, usa wheel per zoom, trascina lo sfondo per pan, "
        "apri i dettagli del nodo per vedere porte, eventi e parametri correnti."
    )


def render_composer_config_editor() -> None:
    generated_config = generated_pipeline_config_dict()
    generated_text = config_to_text(generated_config)
    raw_key = session.PIPELINE_CONFIG_EDITOR_RAW
    baseline_key = session.PIPELINE_CONFIG_EDITOR_BASELINE
    widget_key = session.PIPELINE_CONFIG_EDITOR_WIDGET
    previous_baseline = st.session_state.get(baseline_key)

    current_text, current_baseline = sync_editor_text(
        st.session_state.get(raw_key),
        previous_baseline,
        generated_text,
    )
    session.put(raw_key, current_text)
    session.put(baseline_key, current_baseline)

    widget_value = st.session_state.get(widget_key)
    if should_refresh_editor_widget(widget_value, previous_baseline):
        st.session_state[widget_key] = current_text

    override_enabled = has_manual_config_override()
    if override_enabled:
        st.warning("Override JSON attivo: Run usa il JSON applicato, non i controlli del composer.")

    restore_label = "Disattiva override JSON e usa controlli composer" if override_enabled else "Sincronizza dai controlli composer"
    if st.button(restore_label, width="stretch"):
        session.put(raw_key, generated_text)
        session.put(baseline_key, generated_text)
        session.put(widget_key, generated_text)
        session.put(session.PIPELINE_CONFIG, generated_config)
        session.put(session.PIPELINE_CONFIG_OVERRIDE_ENABLED, False)
        st.rerun()

    st.caption("Modifica il JSON qui sotto solo se vuoi un override esplicito della configurazione generata dai controlli.")
    edited_text = st.text_area(
        "Config JSON modificabile",
        key=widget_key,
        height=420,
    )
    session.put(raw_key, edited_text)

    apply_clicked = st.button("Applica override JSON", type="primary", width="stretch")
    try:
        edited_config = parse_config_text(edited_text)
        if apply_clicked:
            session.put(session.PIPELINE_CONFIG, edited_config)
            session.put(session.PIPELINE_CONFIG_OVERRIDE_ENABLED, True)
            st.success("Override JSON applicato. Run usera questa versione.")
            st.rerun()
        elif override_enabled:
            st.info("JSON valido. Run usa ancora l'ultimo override applicato.")
        else:
            st.success("JSON valido. Run usa i controlli composer finche non applichi l'override.")
    except Exception as exc:
        st.error(f"JSON non valido: {exc}")
        st.warning("Run usera i controlli composer o l'ultimo override valido gia applicato.")
        if session.get(session.PIPELINE_CONFIG) is None:
            session.put(session.PIPELINE_CONFIG, generated_config)


def render_event_integration_editor(registry) -> None:
    st.markdown("### Event & Branching")
    status = event_integration_status()
    branching_options = _plugin_names(registry, PluginCategory.BRANCHING_RULE)

    c1, c2, c3 = st.columns(3)
    c1.metric("Lifecycle bus", "enabled" if status["lifecycle_bus"] else "idle")
    c2.metric("Domain bus", "enabled" if status["domain_bus"] else "idle")
    c3.metric("Branching rules", len(status["branching_rules"]))

    with st.expander("Lifecycle & domain channels", expanded=False):
        st.caption("Il core gia espone lifecycle events del runner e domain events dei componenti `IEventEmitter`.")
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
                context = context_from_config(current_pipeline_config_dict(), registry)
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
            format_func=_display_plugin(registry, PluginCategory.BRANCHING_RULE),
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


def render_log_terminal(*, key_prefix: str) -> None:
    """Render a bounded terminal-like log viewer with level filtering."""
    st.markdown("### Terminale log")
    c1, c2, c3 = st.columns([0.35, 0.35, 0.30])
    show_logs = c1.toggle("Mostra log", key=f"sef_show_log_terminal_{key_prefix}")
    selected_level = c2.selectbox(
        "Livello",
        available_log_levels(),
        index=1,
        key=f"sef_log_terminal_level_{key_prefix}",
    )
    set_capture_level(selected_level)
    if c3.button("Clear logs", width="stretch", key=f"sef_clear_logs_{key_prefix}"):
        clear_log_records()
        st.rerun()

    if not show_logs:
        st.caption("Attiva la visualizzazione per ispezionare i log Python di `library` e `ui`.")
        return

    records = log_records(selected_level)
    if not records:
        st.info("Nessun log per il livello selezionato.")
        return

    st.code("\n".join(record.as_terminal_line() for record in records[-160:]), language="text")


def render_log_tail() -> None:
    """Render recent logs without adding another level controller."""
    selected_level = str(st.session_state.get("sef_log_terminal_level_plan", "INFO"))
    records = log_records(selected_level)
    if not records:
        st.info("Nessun log recente per il livello selezionato nella tab Plan & Logs.")
        return
    st.code("\n".join(record.as_terminal_line() for record in records[-80:]), language="text")


def render_execution(registry) -> None:
    st.subheader("Run & Monitor")
    config = current_pipeline_config_dict()
    issues = validate_runtime_requirements(config)
    plan_preview = build_execution_plan_preview(config, registry)
    plan_summary = summarize_execution_plan(plan_preview.plan) if plan_preview.plan else None
    main_thread_visualizers = main_thread_visualizer_names(config, registry)
    browser_realtime_visualizers = streamlit_realtime_visualizer_names(config)

    status_cols = st.columns(4)
    status_cols[0].metric("Active", len(active_ids()))
    status_cols[1].metric("Snapshots", len(snapshots()))
    status_cols[2].metric("Events", len(event_records()))
    status_cols[3].metric("Analyzers", len(config.get("pipeline", {}).get("analyzers", [])))

    if plan_summary is not None:
        plan_cols = st.columns(4)
        plan_cols[0].metric("Streaming stages", plan_summary["streaming_count"])
        plan_cols[1].metric("Batch stages", plan_summary["batch_count"])
        plan_cols[2].metric("Materializations", plan_summary["materialization_count"])
        plan_cols[3].metric("Parallel-capable", plan_summary["parallel_count"])
    elif plan_preview.error:
        st.warning(f"Planner non disponibile: {plan_preview.error}")

    if issues:
        for issue in issues:
            st.warning(issue)
    if has_manual_config_override():
        st.info("Run sta usando l'override JSON della tab Composer, non la config generata dai controlli.")
    if main_thread_visualizers:
        st.warning(
            "I visualizer live OpenCV aprono finestre native e non sono supportati nel runtime Streamlit. "
            f"Selezionati: {', '.join(main_thread_visualizers)}."
        )
    if browser_realtime_visualizers:
        st.info(
            "Preview realtime browser attiva: usa Submit async per vedere i frame mentre la pipeline gira. "
            f"Selezionati: {', '.join(browser_realtime_visualizers)}."
        )

    run_width_percent = st.slider(
        "Larghezza Esecuzione / Preview",
        min_value=25,
        max_value=75,
        value=int(st.session_state.get("sef_run_panel_width_percent", 38)),
        step=5,
        key="sef_run_panel_width_percent",
        help="Regola lo spazio orizzontale tra esecuzione/preview live e monitor/status.",
    )
    col_run, col_monitor = st.columns([run_width_percent, 100 - run_width_percent], gap="large")
    with col_run:
        st.markdown("### Esecuzione")
        pipeline_id = st.text_input(
            "Pipeline ID",
            value=f"ui-{uuid.uuid4().hex[:8]}",
            key="sef_run_pipeline_id",
        )
        worker_count = st.number_input(
            "Background workers",
            min_value=1,
            max_value=8,
            value=runner_parallelism(),
            key="sef_runner_max_workers",
            help="Numero massimo di pipeline async eseguibili in parallelo.",
        )
        ok, message = configure_runner(int(worker_count))
        if not ok:
            st.warning(message)
        sync_clicked = st.button(
            "Run sync",
            type="primary",
            width="stretch",
            disabled=bool(issues) or bool(browser_realtime_visualizers),
        )
        async_clicked = st.button(
            "Submit async",
            width="stretch",
            disabled=bool(issues) or bool(main_thread_visualizers),
        )

        if sync_clicked:
            execute_sync(registry, config, pipeline_id)
        if async_clicked:
            execute_async(registry, config, pipeline_id)

        current_pipeline_id = session.get(session.PIPELINE_OUTPUT_PIPELINE_ID)
        preview_pipeline_id = _realtime_preview_pipeline_id(current_pipeline_id, pipeline_id)
        if browser_realtime_visualizers or preview_has_content(preview_pipeline_id):
            render_realtime_preview(preview_pipeline_id, title="Preview realtime")
        render_stored_outputs_browser(default_pipeline_id=current_pipeline_id)

    with col_monitor:
        controls_col1, controls_col2 = st.columns(2)
        if controls_col1.button("Refresh", width="stretch", key="sef_refresh_monitor"):
            st.rerun()
        if controls_col2.button("Clear events", width="stretch", key="sef_clear_events"):
            clear_event_records()
            st.rerun()

        render_pipeline_status_dashboard(snapshots(), event_records(), title="Pipeline status")
        render_event_timeline(event_records())
        with st.expander("Execution plan", expanded=False):
            if plan_preview.plan is not None:
                render_execution_plan(plan_preview.plan, title=None)
            elif plan_preview.error:
                st.error(plan_preview.error)
        with st.expander("Terminale log", expanded=False):
            render_log_tail()
        with st.expander("Controls", expanded=False):
            active = active_ids()
            if active:
                st.markdown("**Cancel best-effort**")
                for pipeline_id in active:
                    if st.button(f"Cancel {pipeline_id}", key=f"cancel_{pipeline_id}"):
                        cancelled = cancel_async(pipeline_id)
                        st.toast("Cancelled queued pipeline." if cancelled else "Pipeline already running or unknown.")
                        st.rerun()


def render_stored_outputs_browser(*, default_pipeline_id: str | None = None) -> None:
    available_ids = [snapshot.pipeline_id for snapshot in snapshots() if pipeline_outputs(snapshot.pipeline_id) is not None]
    if not available_ids:
        return

    st.markdown("### Stored outputs")
    selection_key = "sef_stored_output_pipeline_id"
    candidate = session.get(selection_key)
    if default_pipeline_id in available_ids and candidate != default_pipeline_id:
        candidate = default_pipeline_id
        st.session_state[selection_key] = candidate
    elif candidate not in available_ids:
        candidate = default_pipeline_id if default_pipeline_id in available_ids else available_ids[-1]
        st.session_state[selection_key] = candidate

    selected_pipeline_id = st.selectbox(
        "Inspect pipeline outputs",
        available_ids,
        index=available_ids.index(st.session_state[selection_key]),
        key=selection_key,
    )
    outputs = pipeline_outputs(selected_pipeline_id)
    if outputs is not None:
        render_pipeline_outputs(outputs, title=selected_pipeline_id)


def render_registry(registry) -> None:
    st.subheader("Registry operativo")
    st.caption("Componenti disponibili e registrazione runtime di nuovi plugin.")

    categories = [
        PluginCategory.FRAME_EXTRACTOR,
        PluginCategory.SINGLE_FRAME_PROCESSOR,
        PluginCategory.FRAME_BUFFER_PROCESSOR,
        PluginCategory.SIGNAL_EXTRACTOR,
        PluginCategory.SIGNAL_CLEANER,
        PluginCategory.ANALYZER,
        PluginCategory.VISUALIZER,
        PluginCategory.BRANCHING_RULE,
    ]

    selected_category = st.selectbox(
        "Categoria",
        categories,
        format_func=lambda category: category.value.replace("_", " ").title(),
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
            format_func=lambda item: item.value.replace("_", " ").title(),
        )
        name = st.text_input("Nome registry", placeholder="my_custom_analyzer")
        class_path = st.text_input("Classe Python", placeholder="my_package.module.MyAnalyzer")
        description = st.text_input("Descrizione")
        submitted = st.form_submit_button("Registra componente")

    if submitted:
        register_runtime_plugin(registry, category, name, class_path, description)


def render_config_lab(registry) -> None:
    st.subheader("Config lab")
    current_config = current_pipeline_config_dict()
    raw_config = st.text_area(
        "Config JSON",
        value=config_to_text(current_config),
        height=440,
    )

    c1, c2 = st.columns(2)
    if c1.button("Validate config", width="stretch"):
        try:
            context = context_from_config(json.loads(raw_config), registry)
            st.success(
                "Config valida: "
                f"{len(context.frame_processors)} frame processor, "
                f"{len(context.signal_cleaners)} signal cleaner, "
                f"{len(context.analyzers)} analyzer."
            )
        except Exception as exc:
            st.error(f"Config non valida: {exc}")

    if c2.button("Run config", type="primary", width="stretch"):
        try:
            config = json.loads(raw_config)
            _raise_runtime_issues(config)
            pipeline_id = f"config-{uuid.uuid4().hex[:8]}"
            execution_config = _prepare_execution_config(config, pipeline_id)
            context = context_from_config(execution_config, registry)
            outputs = run_sync(context, pipeline_id=pipeline_id)
            session.put(session.PIPELINE_OUTPUT_PIPELINE_ID, pipeline_id)
            st.success(f"Pipeline completata: {len(outputs.results)} risultati, {outputs.artifact_count} artifact.")
            render_pipeline_outputs(outputs, title="Config outputs")
        except Exception as exc:
            st.error(f"Esecuzione fallita: {exc}")

    if st.button("Submit async config", width="stretch"):
        try:
            config = json.loads(raw_config)
            _raise_runtime_issues(config)
            pipeline_id = f"config-{uuid.uuid4().hex[:8]}"
            execution_config = _prepare_execution_config(config, pipeline_id)
            context = context_from_config(execution_config, registry)
            submitted_id = submit_async(pipeline_id, context)
            session.put(session.PIPELINE_OUTPUT_PIPELINE_ID, submitted_id)
            st.success(f"Pipeline {submitted_id} sottomessa in background.")
            time.sleep(0.2)
            st.rerun()
        except Exception as exc:
            st.error(f"Submit config fallito: {exc}")

    preview_pipeline_id = session.get(session.REALTIME_PREVIEW_PIPELINE_ID)
    if preview_pipeline_id and preview_has_content(preview_pipeline_id):
        render_realtime_preview(preview_pipeline_id, title="Preview realtime config")


def execute_sync(registry, config: dict[str, Any], pipeline_id: str) -> None:
    try:
        session.clear(session.TRACKING_VIDEO_CACHE)
        session.clear(session.PIPELINE_OUTPUTS)
        execution_config = _prepare_execution_config(config, pipeline_id)
        context = context_from_config(execution_config, registry)
        with st.spinner("Pipeline in esecuzione..."):
            outputs = run_sync(context, pipeline_id=pipeline_id)
        session.put(session.PIPELINE_OUTPUT_PIPELINE_ID, pipeline_id)
        st.success(f"Pipeline completata: {len(outputs.results)} risultati, {outputs.artifact_count} artifact.")
    except Exception as exc:
        st.error(f"Pipeline fallita: {exc}")


def execute_async(registry, config: dict[str, Any], pipeline_id: str) -> None:
    try:
        session.clear(session.TRACKING_VIDEO_CACHE)
        session.clear(session.PIPELINE_OUTPUTS)
        execution_config = _prepare_execution_config(config, pipeline_id)
        context = context_from_config(execution_config, registry)
        submitted_id = submit_async(pipeline_id, context)
        session.put(session.PIPELINE_OUTPUT_PIPELINE_ID, submitted_id)
        st.success(f"Pipeline {submitted_id} sottomessa in background.")
        time.sleep(0.2)
        st.rerun()
    except Exception as exc:
        st.error(f"Submit fallita: {exc}")


def _prepare_execution_config(config: dict[str, Any], pipeline_id: str) -> dict[str, Any]:
    if not config_has_streamlit_realtime_visualizer(config):
        return config
    reset_sink(pipeline_id)
    session.put(session.REALTIME_PREVIEW_PIPELINE_ID, pipeline_id)
    return with_realtime_sink_ids(config, pipeline_id)


def _raise_runtime_issues(config: dict[str, Any]) -> None:
    issues = validate_runtime_requirements(config)
    if issues:
        raise ValueError("Config non eseguibile dalla UI: " + " ".join(issues))


def _realtime_preview_pipeline_id(current_pipeline_id: str | None, fallback_pipeline_id: str) -> str:
    preview_pipeline_id = session.get(session.REALTIME_PREVIEW_PIPELINE_ID)
    if isinstance(preview_pipeline_id, str) and preview_pipeline_id:
        return preview_pipeline_id
    if isinstance(current_pipeline_id, str) and current_pipeline_id:
        return current_pipeline_id
    return fallback_pipeline_id


def register_runtime_plugin(registry, category: PluginCategory, name: str, class_path: str, description: str) -> None:
    if not name.strip():
        st.error("Il nome registry e obbligatorio.")
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


def _plugin_names(registry, category: PluginCategory) -> list[str]:
    return [plugin.name for plugin in sorted(registry.list(category), key=lambda item: item.name)]


def _display_plugin(registry, category: PluginCategory):
    def _format(name: str) -> str:
        try:
            plugin = registry.get(category, name)
            return f"{plugin.name} - {plugin.factory.__name__}"
        except Exception:
            return name

    return _format


if __name__ == "__main__":
    main()
