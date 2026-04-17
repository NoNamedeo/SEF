"""
Pipeline Monitor — gestisci pipeline in esecuzione asincrona.

Usa ThreadedPipelineRunner per sottomettere pipeline in background,
InMemoryPipelineMonitor per tracciare gli ID attivi e cancellare esecuzioni.

Nota: il wiring event-driven (PipelineOrchestrator + PipelineEvent) è
in sviluppo — questa pagina usa il runner direttamente via pipeline_service.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.services.registry_bootstrap import get_registry  # noqa: E402
from ui.services.pipeline_service import (  # noqa: E402
    submit_async,
    cancel_async,
    active_ids,
    snapshots,
    event_records,
    clear_event_records,
    context_from_config,
    pipeline_outputs,
)
from ui.state import session  # noqa: E402
from ui.components.pipeline_status_dashboard import (  # noqa: E402
    render_event_timeline,
    render_pipeline_status_dashboard,
)
from ui.components.pipeline_outputs_viewer import render_pipeline_outputs  # noqa: E402

st.set_page_config(page_title="Pipeline Monitor — SEF", layout="wide", page_icon="📊")
st.title("📊 Pipeline Monitor")
st.caption("Sottometti pipeline in background con ThreadedPipelineRunner, monitora gli ID attivi e cancella esecuzioni.")

registry = get_registry()

render_pipeline_status_dashboard(
    snapshots=snapshots(),
    events=event_records(),
    title="Pipeline status",
)
render_event_timeline(event_records())

with st.expander("Controls", expanded=False):
    c1, c2 = st.columns(2)
    if c1.button("Refresh", width="stretch"):
        st.rerun()
    if c2.button("Clear events", width="stretch"):
        clear_event_records()
        st.rerun()

    ids = active_ids()
    if ids:
        st.markdown("**Cancel best-effort**")
        for pid in ids:
            if st.button(f"Cancel {pid}", key=f"cancel_{pid}"):
                cancel_async(pid)
                st.success(f"Richiesta di cancellazione inviata per `{pid}`.")
                st.rerun()

# ── Submit new pipeline ───────────────────────────────────────────────────────
st.subheader("Sottometti nuova pipeline")

st.caption("Usa la configurazione salvata nel **Config Builder** oppure incolla una nuova configurazione JSON.")

import json

saved_cfg = session.get(session.PIPELINE_CONFIG)
default_json = (
    json.dumps(saved_cfg, indent=2)
    if saved_cfg
    else json.dumps(
        {
            "pipeline": {
                "frame_extractor": {
                    "name": "opencv_buffered",
                    "params": {"path": str(_ROOT / "videos" / "Traffic.mp4"), "resize": [640, 480], "stride": 3, "max_frames": 200},
                },
                "frame_cleaners": [{"name": "smoothing"}],
                "signal_extractor": {
                    "name": "opencv_tracker",
                    "params": {"tracker_type": "CSRT", "start_box": [260, 160, 120, 160]},
                },
                "signal_cleaners": [{"name": "moving_average", "params": {"window_size": 5}}],
                "analyzers": [{"name": "vertical_position"}],
                "visualizers": [],
            }
        },
        indent=2,
    )
)

col_form, col_info = st.columns([2, 1])

with col_form:
    raw = st.text_area("Config JSON", value=default_json, height=320, key="mon_cfg_text")
    pipeline_id_input = st.text_input(
        "Pipeline ID (lascia vuoto per generare automaticamente)",
        key="mon_pid",
    )

    submit_clicked = st.button("Sottometti in background", type="primary", width="stretch")

with col_info:
    st.markdown("**Come funziona**")
    st.markdown("""
    1. Il context viene creato dal config JSON e affidato al `PipelineOrchestrator`.
    2. L'esecuzione avviene su un thread separato: la UI rimane reattiva.
    3. Lo stato viene tracciato da `InMemoryPipelineMonitor`.
    4. Puoi cancellare una pipeline attiva dal pannello **Controls**.

    > Gli output completati vengono persistiti nello store applicativo e
    > possono essere ispezionati direttamente da questa pagina.
    """)

    st.markdown("**Architettura**")
    st.code(
        """
PipelineOrchestrator
  .submit(context, id)
    → ThreadedPipelineRunner
      → Pipeline.run()
      → monitor.complete(id)
    """,
        language="text",
    )

if submit_clicked:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        st.error(f"JSON non valido: {exc}")
        st.stop()

    try:
        ctx = context_from_config(parsed, registry)
    except Exception as exc:
        st.error(f"Costruzione contesto fallita: {exc}")
        st.stop()

    pid = pipeline_id_input.strip() or f"pipeline-{uuid.uuid4().hex[:8]}"
    try:
        submit_async(pid, ctx)
        st.success(f"Pipeline `{pid}` sottomessa. Aggiorna il monitor per vedere lo stato.")
        time.sleep(0.3)  # give the thread a moment to register
        st.rerun()
    except Exception as exc:
        st.error(f"Errore nella sottomissione: {exc}")

st.divider()

# ── Coming soon ───────────────────────────────────────────────────────────────
with st.expander("🚧 Funzionalità in sviluppo", expanded=False):
    st.markdown("""
    Le seguenti funzionalità sono basate su contratti già definiti nella libreria
    e saranno disponibili non appena il wiring sarà completato:

    | Funzionalità | Contratto | Stato |
    |---|---|---|
    | Trigger event-driven | `PipelineOrchestrator` + `PipelineEvent` | ✅ Disponibile |
    | Pipeline condizionali | `BranchingCoordinator` + `IBranchingRule` | ✅ Disponibile |
    | Retry configurabili | `FixedRetryPolicy` / `ExponentialBackoffRetryPolicy` | ✅ Disponibile oggi |
    | Lifecycle events | `EventBus` + `PipelineLifecycleEvent` | ✅ Disponibile oggi |
    | Output persistiti async | `IPipelineOutputStore` | ✅ Disponibile |
    """)


def render_stored_outputs_browser() -> None:
    available_ids = [
        snapshot.pipeline_id
        for snapshot in snapshots()
        if pipeline_outputs(snapshot.pipeline_id) is not None
    ]
    if not available_ids:
        return

    st.markdown("### Stored outputs")
    selected_pipeline_id = st.selectbox(
        "Inspect pipeline outputs",
        available_ids,
        index=len(available_ids) - 1,
        key="monitor_stored_output_pipeline_id",
    )
    outputs = pipeline_outputs(selected_pipeline_id)
    if outputs is not None:
        render_pipeline_outputs(outputs, title=selected_pipeline_id)


render_stored_outputs_browser()
