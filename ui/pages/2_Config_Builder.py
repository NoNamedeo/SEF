"""
Config Builder — costruisci una pipeline tramite dizionario di configurazione.

Il formato segue lo schema di ConfigPipelineBuilder:

  pipeline:
    frame_extractor:
      name: opencv_buffered
      params:
        path: "/abs/path/to/video.mp4"
        resize: [640, 480]
        stride: 2
    frame_cleaners:
      - name: smoothing
    signal_extractor:
      name: opencv_tracker
      params:
        tracker_type: CSRT
        start_box: [100, 200, 80, 60]
    signal_cleaners:
      - name: moving_average
        params:
          window_size: 5
    analyzers:
      - name: vertical_position
    visualizers: []
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.services.registry_bootstrap import get_registry  # noqa: E402
from ui.services.pipeline_service import run_sync, context_from_config  # noqa: E402
from ui.components.results_viewer import render_results  # noqa: E402
from ui.state import session  # noqa: E402

st.set_page_config(page_title="Config Builder — SEF", layout="wide", page_icon="⚙️")
st.title("⚙️ Config Builder")
st.caption("Definisci la pipeline tramite un dizionario di configurazione. Il registry risolve i nomi nei componenti concreti.")

registry = get_registry()

# ── Default config template ───────────────────────────────────────────────────
_DEMO_VIDEO = str(_ROOT / "videos" / "Traffic.mp4")

_DEFAULT_CONFIG = {
    "pipeline": {
        "frame_extractor": {
            "name": "opencv_buffered",
            "params": {"path": _DEMO_VIDEO, "resize": [640, 480], "stride": 2, "max_frames": 300},
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
}

# ── Layout ────────────────────────────────────────────────────────────────────
col_editor, col_preview = st.columns([1, 1], gap="large")

with col_editor:
    st.markdown("### Configurazione")

    # Populate editor from session or default
    saved_cfg = session.get(session.PIPELINE_CONFIG) or _DEFAULT_CONFIG
    raw_text = st.text_area(
        "Config JSON",
        value=json.dumps(saved_cfg, indent=2),
        height=500,
        key="cb_raw_text",
    )

    col_validate, col_reset = st.columns(2)
    validate_clicked = col_validate.button("🔍 Valida", width="stretch")
    if col_reset.button("↺ Reset default", width="stretch"):
        session.put(session.PIPELINE_CONFIG, _DEFAULT_CONFIG)
        st.rerun()

with col_preview:
    st.markdown("### Componenti che verranno creati")

    parsed_cfg: dict | None = None
    parse_error: str | None = None

    try:
        parsed_cfg = json.loads(raw_text)
        session.put(session.PIPELINE_CONFIG, parsed_cfg)
    except json.JSONDecodeError as exc:
        parse_error = str(exc)

    if parse_error:
        st.error(f"JSON non valido: {parse_error}")
    elif parsed_cfg:
        cfg = parsed_cfg.get("pipeline", parsed_cfg)

        def _preview_entry(label: str, entry: dict | None) -> None:
            if not entry:
                st.write(f"- **{label}**: _non configurato_")
                return
            name = entry.get("name", "?")
            params = entry.get("params", {})
            try:
                defn = registry.get(label.lower().replace(" ", "_"), name)
                st.write(f"- **{label}**: `{name}` ✅  \n  _{defn.description}_")
            except KeyError:
                st.write(f"- **{label}**: `{name}` ⚠️ _plugin non trovato nel registry_")
            if params:
                st.json(params, expanded=False)

        def _preview_list(label: str, entries: list[dict]) -> None:
            if not entries:
                st.write(f"- **{label}**: _nessuno_")
                return
            for e in entries:
                _preview_entry(label, e)

        from library.core.plugins.PluginRegistry import PluginCategory as PC

        _preview_entry("frame_extractor", cfg.get("frame_extractor"))
        _preview_list("frame_cleaner", cfg.get("frame_cleaners", []))
        _preview_entry("signal_extractor", cfg.get("signal_extractor"))
        _preview_list("signal_cleaner", cfg.get("signal_cleaners", []))
        _preview_list("analyzer", cfg.get("analyzers", []))
        _preview_list("visualizer", cfg.get("visualizers", []))

# ── Validate ──────────────────────────────────────────────────────────────────
if validate_clicked:
    if parse_error:
        st.error("Correggi gli errori di JSON prima di validare.")
    else:
        try:
            ctx = context_from_config(parsed_cfg, registry)
            st.success(
                f"✅ Configurazione valida — "
                f"{len(ctx.frame_cleaners)} cleaner(s), "
                f"{len(ctx.signal_cleaners)} signal cleaner(s), "
                f"{len(ctx.analyzers)} analizzatore/i."
            )
        except Exception as exc:
            st.error(f"Errore nella costruzione del contesto: {exc}")

st.divider()

# ── Run ───────────────────────────────────────────────────────────────────────
st.markdown("### Esecuzione")

run_disabled = parse_error is not None or parsed_cfg is None
if st.button("Costruisci ed esegui", type="primary", width="stretch", disabled=run_disabled):
    try:
        ctx = context_from_config(parsed_cfg, registry)
    except Exception as exc:
        st.error(f"Costruzione contesto fallita: {exc}")
        st.stop()

    with st.spinner("Pipeline in esecuzione…"):
        try:
            results = run_sync(ctx)
            session.put(session.PIPELINE_RESULTS, results)
            st.success(f"Pipeline completata — {len(results)} risultato/i.")
        except Exception as exc:
            st.error(f"Pipeline fallita: {exc}")
            st.stop()

results = session.get(session.PIPELINE_RESULTS)
if results:
    st.divider()
    render_results(results)
