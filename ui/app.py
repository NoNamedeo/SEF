"""
SEF Studio.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import importlib
import html
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

from sef.core.events.PipelineLifecycleEvent import PipelineLifecycleEvent  # noqa: E402
from sef.core.plugins.PluginRegistry import PluginCategory  # noqa: E402
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
    SCENARIO_OPTIONS,
    apply_aruco_preset,
    apply_tracking_preset,
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
from ui.services.plugin_display import plugin_factory_label  # noqa: E402
from ui.services.realtime_preview_service import (  # noqa: E402
    config_has_streamlit_realtime_visualizer,
    preview_has_content,
    reset_sink,
    streamlit_realtime_visualizer_names,
    with_realtime_sink_ids,
)
from ui.services.registry_catalog import (  # noqa: E402
    RegistryPluginCard,
    build_registry_catalog,
    filter_registry_cards,
    metadata_as_pretty_json,
    registry_category_label,
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
        if st.button("ArUco marker motion", width="stretch"):
            apply_aruco_preset()
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
        list(SCENARIO_OPTIONS),
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
        st.caption("Attiva la visualizzazione per ispezionare i log Python di `sef` e `ui`.")
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
    st.caption("Catalogo live dei plugin SEF: categorie, metadata, aliases, factory path, capability e dipendenze opzionali.")

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
    catalog = build_registry_catalog(registry)
    _inject_registry_styles()

    summary_cols = st.columns(4)
    summary_cols[0].metric("Plugin", len(catalog.cards))
    summary_cols[1].metric("Categorie", len(catalog.categories))
    summary_cols[2].metric("Tag", len(catalog.tags))
    summary_cols[3].metric("Extras opzionali", len(catalog.optional_extras))

    search_col, category_col = st.columns([0.45, 0.55])
    query = search_col.text_input(
        "Cerca nel registry",
        placeholder="nome, tag, optional extra, factory path, metadata...",
        key="sef_registry_search",
    )
    selected_categories = category_col.multiselect(
        "Categorie",
        [category.value for category in categories],
        format_func=registry_category_label,
        key="sef_registry_category_filter",
    )

    tag_col, extra_col, capability_col = st.columns(3)
    selected_tags = tag_col.multiselect("Tag", list(catalog.tags), key="sef_registry_tag_filter")
    selected_extras = extra_col.multiselect("Optional extra", list(catalog.optional_extras), key="sef_registry_extra_filter")
    selected_capabilities = capability_col.multiselect(
        "Capability vere",
        list(catalog.capability_names),
        key="sef_registry_capability_filter",
    )

    filtered_cards = filter_registry_cards(
        catalog.cards,
        query=query,
        categories=selected_categories,
        tags=selected_tags,
        optional_extras=selected_extras,
        capabilities=selected_capabilities,
    )

    view_mode = st.segmented_control(
        "Vista registry",
        ["Cards", "Table", "Metadata"],
        default="Cards",
        key="sef_registry_view_mode",
    )
    st.caption(f"{len(filtered_cards)} plugin mostrati su {len(catalog.cards)}.")

    if view_mode == "Table":
        st.dataframe([card.table_row() for card in filtered_cards], hide_index=True, width="stretch")
    elif view_mode == "Metadata":
        _render_registry_metadata_view(filtered_cards)
    else:
        _render_registry_cards(filtered_cards)

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
        c1, c2 = st.columns(2)
        version = c1.text_input("Versione", value="1.0.0")
        aliases_text = c2.text_input("Aliases", placeholder="alias_a, alias_b")
        metadata_text = st.text_area(
            "Metadata JSON",
            value='{\n  "tags": ["custom"],\n  "domain": "experiment"\n}',
            height=150,
            help=(
                "Metadati suggeriti: tags, domain, owner, optional_extra, expected_input, "
                "expected_output, hardware, maturity, docs_url."
            ),
        )
        submitted = st.form_submit_button("Registra componente")

    if submitted:
        register_runtime_plugin(
            registry,
            category,
            name,
            class_path,
            description,
            version=version,
            aliases=tuple(alias.strip() for alias in aliases_text.split(",") if alias.strip()),
            metadata_text=metadata_text,
        )


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


def register_runtime_plugin(
    registry,
    category: PluginCategory,
    name: str,
    class_path: str,
    description: str,
    *,
    version: str,
    aliases: tuple[str, ...],
    metadata_text: str,
) -> None:
    if not name.strip():
        st.error("Il nome registry e obbligatorio.")
        return
    if "." not in class_path:
        st.error("Inserisci un path completo del tipo package.module.ClassName.")
        return
    try:
        metadata = json.loads(metadata_text) if metadata_text.strip() else {}
    except json.JSONDecodeError as exc:
        st.error(f"Metadata JSON non valido: {exc}")
        return
    if not isinstance(metadata, dict):
        st.error("Metadata JSON deve essere un oggetto.")
        return
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        factory = getattr(module, class_name)
        registry.register(
            category,
            name.strip(),
            factory,
            description.strip(),
            version=version.strip() or "1.0.0",
            aliases=aliases,
            metadata=metadata,
        )
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
            return f"{plugin.name} - {plugin_factory_label(plugin)}"
        except Exception:
            return name

    return _format


def _render_registry_cards(cards: tuple[RegistryPluginCard, ...]) -> None:
    if not cards:
        st.info("Nessun plugin corrisponde ai filtri correnti.")
        return

    for category in sorted({card.category for card in cards}):
        group = tuple(card for card in cards if card.category == category)
        style = _registry_category_style(category)
        st.markdown(
            (
                f"<div class='sef-category-heading' style='border-color:{style['accent']};'>"
                f"<span style='background:{style['accent']};'></span>"
                f"{html.escape(registry_category_label(category))}"
                f"<small>{len(group)} plugin</small>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        columns = st.columns(2)
        for index, card in enumerate(group):
            with columns[index % 2]:
                _render_registry_card(card)


def _render_registry_card(card: RegistryPluginCard) -> None:
    style = _registry_category_style(card.category)
    tags = "".join(f"<span class='sef-chip'>{html.escape(tag)}</span>" for tag in card.tags[:8])
    aliases = ", ".join(card.aliases) if card.aliases else "-"
    optional_extra = card.optional_extra or "core"
    capabilities = _capability_badges(card)
    metadata_preview = metadata_as_pretty_json(card.metadata) if card.metadata else "{}"
    st.markdown(
        f"""
        <div class="sef-registry-card" style="border-top-color:{style['accent']}; background:{style['background']};">
          <div class="sef-registry-card-title">
            <strong>{html.escape(card.name)}</strong>
            <span style="background:{style['accent']};">{html.escape(card.category_label)}</span>
          </div>
          <p>{html.escape(card.description or "Nessuna descrizione disponibile.")}</p>
          <div class="sef-registry-meta">
            <span>version <b>{html.escape(card.version)}</b></span>
            <span>extra <b>{html.escape(optional_extra)}</b></span>
            <span>aliases <b>{html.escape(aliases)}</b></span>
          </div>
          <div class="sef-registry-tags">{tags or "<span class='sef-muted'>nessun tag</span>"}</div>
          <div class="sef-registry-capabilities">{capabilities}</div>
          <details>
            <summary>Factory path</summary>
            <code>{html.escape(card.factory_path)}</code>
          </details>
          <details>
            <summary>Metadata</summary>
            <pre>{html.escape(metadata_preview)}</pre>
          </details>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_registry_metadata_view(cards: tuple[RegistryPluginCard, ...]) -> None:
    if not cards:
        st.info("Nessun metadata disponibile per i filtri correnti.")
        return
    selected_name = st.selectbox(
        "Plugin",
        [f"{card.category}/{card.name}" for card in cards],
        key="sef_registry_metadata_plugin",
    )
    selected_card = cards[[f"{card.category}/{card.name}" for card in cards].index(selected_name)]
    c1, c2 = st.columns([0.42, 0.58])
    with c1:
        st.markdown("**Descriptor registry**")
        st.json(
            {
                "category": selected_card.category,
                "name": selected_card.name,
                "version": selected_card.version,
                "aliases": list(selected_card.aliases),
                "factory_path": selected_card.factory_path,
                "description": selected_card.description,
                "tags": list(selected_card.tags),
                "optional_extra": selected_card.optional_extra,
            }
        )
    with c2:
        st.markdown("**Metadata e capability**")
        st.json(
            {
                "metadata": dict(selected_card.metadata),
                "capabilities": dict(selected_card.capabilities),
            }
        )


def _capability_badges(card: RegistryPluginCard) -> str:
    if not card.capabilities:
        return "<span class='sef-muted'>capabilities non dichiarate sul factory lazy</span>"
    badges = []
    for name, enabled in sorted(card.capabilities.items()):
        state_class = "sef-cap-on" if enabled else "sef-cap-off"
        badges.append(f"<span class='{state_class}'>{html.escape(name)}</span>")
    return "".join(badges)


def _inject_registry_styles() -> None:
    st.markdown(
        """
        <style>
        .sef-category-heading {
          display: flex;
          align-items: center;
          gap: 0.55rem;
          border-left: 4px solid;
          padding: 0.45rem 0.7rem;
          margin: 1.1rem 0 0.45rem;
          background: rgba(255,255,255,0.035);
          font-weight: 700;
        }
        .sef-category-heading span {
          width: 0.7rem;
          height: 0.7rem;
          border-radius: 999px;
          display: inline-block;
        }
        .sef-category-heading small {
          color: rgba(250,250,250,0.62);
          font-weight: 500;
          margin-left: auto;
        }
        .sef-registry-card {
          border: 1px solid rgba(255,255,255,0.12);
          border-top: 4px solid;
          border-radius: 8px;
          padding: 0.85rem;
          margin-bottom: 0.85rem;
          min-height: 18rem;
        }
        .sef-registry-card-title {
          display: flex;
          gap: 0.55rem;
          justify-content: space-between;
          align-items: start;
        }
        .sef-registry-card-title strong {
          font-size: 1rem;
          line-height: 1.2;
        }
        .sef-registry-card-title span {
          border-radius: 999px;
          color: #0f1419;
          font-size: 0.72rem;
          font-weight: 800;
          padding: 0.18rem 0.45rem;
          white-space: nowrap;
        }
        .sef-registry-card p {
          color: rgba(250,250,250,0.76);
          min-height: 2.6rem;
          margin: 0.55rem 0;
        }
        .sef-registry-meta,
        .sef-registry-tags,
        .sef-registry-capabilities {
          display: flex;
          flex-wrap: wrap;
          gap: 0.35rem;
          margin: 0.45rem 0;
        }
        .sef-registry-meta span,
        .sef-chip,
        .sef-cap-on,
        .sef-cap-off {
          border-radius: 999px;
          padding: 0.16rem 0.42rem;
          font-size: 0.72rem;
        }
        .sef-registry-meta span {
          background: rgba(255,255,255,0.07);
          color: rgba(250,250,250,0.72);
        }
        .sef-chip {
          background: rgba(125,211,252,0.12);
          color: #bae6fd;
          border: 1px solid rgba(125,211,252,0.22);
        }
        .sef-cap-on {
          background: rgba(134,239,172,0.12);
          color: #bbf7d0;
          border: 1px solid rgba(134,239,172,0.22);
        }
        .sef-cap-off {
          background: rgba(248,113,113,0.10);
          color: #fecaca;
          border: 1px solid rgba(248,113,113,0.18);
        }
        .sef-muted {
          color: rgba(250,250,250,0.54);
          font-size: 0.78rem;
        }
        .sef-registry-card details {
          margin-top: 0.45rem;
        }
        .sef-registry-card summary {
          cursor: pointer;
          color: rgba(250,250,250,0.7);
          font-size: 0.8rem;
        }
        .sef-registry-card code,
        .sef-registry-card pre {
          white-space: pre-wrap;
          word-break: break-word;
          font-size: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _registry_category_style(category: str) -> dict[str, str]:
    styles = {
        PluginCategory.FRAME_EXTRACTOR.value: {"accent": "#7dd3fc", "background": "rgba(14,116,144,0.13)"},
        PluginCategory.SINGLE_FRAME_PROCESSOR.value: {"accent": "#a7f3d0", "background": "rgba(6,95,70,0.12)"},
        PluginCategory.FRAME_BUFFER_PROCESSOR.value: {"accent": "#fcd34d", "background": "rgba(146,64,14,0.12)"},
        PluginCategory.SIGNAL_EXTRACTOR.value: {"accent": "#c4b5fd", "background": "rgba(91,33,182,0.12)"},
        PluginCategory.SIGNAL_CLEANER.value: {"accent": "#fdba74", "background": "rgba(154,52,18,0.12)"},
        PluginCategory.ANALYZER.value: {"accent": "#f0abfc", "background": "rgba(134,25,143,0.12)"},
        PluginCategory.VISUALIZER.value: {"accent": "#93c5fd", "background": "rgba(30,64,175,0.12)"},
        PluginCategory.BRANCHING_RULE.value: {"accent": "#fda4af", "background": "rgba(159,18,57,0.12)"},
    }
    return styles.get(category, {"accent": "#d1d5db", "background": "rgba(75,85,99,0.12)"})


if __name__ == "__main__":
    main()
