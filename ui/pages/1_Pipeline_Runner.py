"""
Pipeline Runner — esecuzione interattiva di una pipeline completa.

Layout:
  [sidebar]   modalità pipeline + configurazione parametri
  [top]       selezione video  (sempre visibile — prerequisito per i tab)
  Tab 1       ROI / Barriere  (canvas interattivo)
  Tab 2       Esecuzione & Risultati
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.components.video_selector   import render_video_selector    # noqa: E402
from ui.components.roi_selector     import render_roi_selector      # noqa: E402
from ui.components.barrier_selector import render_barrier_selector  # noqa: E402
from ui.components.results_viewer   import render_results           # noqa: E402
from ui.services.pipeline_service   import run_sync                 # noqa: E402
from ui.state                       import session                  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pipeline Runner — SEF", layout="wide", page_icon="🚀")
st.title("🚀 Pipeline Runner")
st.caption("Configura ed esegui una pipeline di analisi video in modalità interattiva.")

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — pipeline mode + all configuration parameters
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Configurazione pipeline")

    mode = st.radio(
        "Tipo di tracking",
        ["Singolo oggetto", "Multi oggetto + barriere"],
        key="pr_mode",
    )

    st.divider()
    st.markdown("**Frame extraction**")

    _resize_map = {
        "Originale": None,
        "320×240":   (320, 240),
        "480×360":   (480, 360),
        "640×480":   (640, 480),
        "960×540":   (960, 540),
    }
    resize_opt = st.selectbox("Resize", list(_resize_map.keys()), key="pr_resize_opt")
    resize_val: tuple | None = _resize_map[resize_opt]
    # Store so ROI/barrier selector can use the same resolution.
    session.put("pr_resize", resize_val)

    stride = st.slider("Stride frame", 1, 10, 2, key="pr_stride",
                        help="Analizza 1 frame ogni N.")
    limit = st.checkbox("Limita frame", key="pr_limit_frames")
    if limit:
        st.number_input("Max frame", 10, 5000, 300, 10, key="pr_max_frames")

    st.divider()
    st.markdown("**Frame cleaners**")
    st.checkbox("Smoothing temporale", value=True, key="pr_smoothing")
    st.checkbox("Scala di grigi",      value=False, key="pr_gray")

    st.divider()
    st.markdown("**Signal extraction**")
    st.selectbox("Tracker", ["CSRT", "KCF", "MIL"], key="pr_tracker")
    if mode == "Multi oggetto + barriere":
        st.slider("Max oggetti",             1,    10,   3,    key="pr_max_obj")
        st.slider("Soglia similarità", 0.3, 0.95, 0.6, 0.05, key="pr_sim")

    st.divider()
    st.markdown("**Signal cleaners**")
    ma_disabled = (mode == "Multi oggetto + barriere")
    st.checkbox("Moving average", value=True, key="pr_mavg", disabled=ma_disabled)
    if not ma_disabled and st.session_state.get("pr_mavg", True):
        st.slider("Finestra moving avg", 3, 21, 5, 2, key="pr_window")

    st.divider()
    st.markdown("**Analisi**")
    if mode == "Singolo oggetto":
        st.multiselect(
            "Analizzatori",
            [
                "Posizione verticale",
                "Frequenza verticale",
                "Posizione orizzontale",
                "Velocità verticale",
                "Velocità orizzontale",
                "Frequenza orizzontale",
            ],
            default=["Posizione verticale"],
            key="pr_analyzers",
        )
    else:
        st.info("Barrier Counting Analyzer automatico.")

# ═════════════════════════════════════════════════════════════════════════════
# VIDEO SELECTION — always above the tabs so the frame is ready for the canvas
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("1 · Sorgente video")

video_path, first_frame, meta = render_video_selector()

if video_path:
    session.put(session.VIDEO_PATH,  video_path)
    session.put(session.FIRST_FRAME, first_frame)
    session.put(session.FRAME_META,  meta)
else:
    for k in (session.VIDEO_PATH, session.FIRST_FRAME, session.FRAME_META,
              session.ROI_BOX, session.BARRIERS, session.PIPELINE_RESULTS):
        session.clear(k)

# ── Gate: nothing below renders until a video is selected ────────────────────
if not video_path:
    st.stop()

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# TABS — ROI/barriers and run (frame is guaranteed to exist here)
# ═════════════════════════════════════════════════════════════════════════════
tab_roi, tab_run = st.tabs(["2 · ROI / Barriere", "3 · Esecuzione & Risultati"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB ROI
# ─────────────────────────────────────────────────────────────────────────────
with tab_roi:
    frame      = session.get(session.FIRST_FRAME)
    resize_cfg = session.get("pr_resize")

    st.markdown("#### Selezione ROI")
    roi = render_roi_selector(frame, resize=resize_cfg, key="pr_roi_canvas")
    if roi:
        session.put(session.ROI_BOX, roi)
        x, y, w, h = roi
        st.success(f"ROI: x={x}, y={y}, w={w}, h={h}")

    if mode == "Multi oggetto + barriere":
        st.divider()
        st.markdown("#### Barriere di conteggio")

        n_barriers = st.number_input(
            "Numero di barriere", min_value=1, max_value=6, value=2, key="pr_n_barriers"
        )
        default_names = [chr(ord("A") + i) for i in range(int(n_barriers))]
        raw_names = st.text_input(
            "Nomi barriere (separati da virgola)",
            value=", ".join(default_names),
            key="pr_barrier_names",
        )
        barrier_names = [n.strip() for n in raw_names.split(",") if n.strip()]

        barriers = render_barrier_selector(
            frame,
            barrier_names=barrier_names,
            resize=resize_cfg,
            state_key="pr_barriers",
        )
        if barriers:
            session.put(session.BARRIERS, barriers)

# ─────────────────────────────────────────────────────────────────────────────
# TAB RUN
# ─────────────────────────────────────────────────────────────────────────────
with tab_run:
    roi_box  = session.get(session.ROI_BOX)
    barriers = session.get(session.BARRIERS) or {}

    # ── Pre-run validation ────────────────────────────────────────────────────
    issues: list[str] = []
    if not roi_box:
        issues.append("Nessuna ROI selezionata — disegnala nel tab **ROI / Barriere**.")
    if mode == "Multi oggetto + barriere" and not barriers:
        issues.append("Nessuna barriera definita — disegnala nel tab **ROI / Barriere**.")
    if mode == "Singolo oggetto" and not session.get("pr_analyzers"):
        issues.append("Seleziona almeno un analizzatore nella sidebar.")

    for msg in issues:
        st.warning(msg)

    run_clicked = st.button(
        "▶ Avvia analisi", type="primary", use_container_width=True,
        disabled=bool(issues), key="pr_run_btn",
    )

    if run_clicked:
        _resize   = session.get("pr_resize")
        _stride   = session.get("pr_stride")   or 2
        _maxf     = int(session.get("pr_max_frames")) if session.get("pr_limit_frames") else None
        _tracker  = session.get("pr_tracker")  or "CSRT"
        _window   = session.get("pr_window")   or 5
        _mavg     = session.get("pr_mavg")     if mode == "Singolo oggetto" else False

        try:
            from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
            from library.core.pipeline.PipelineContext import PipelineContext

            frame_cleaners = []
            if session.get("pr_smoothing"):
                from library.frame_cleaners.SmoothingFrameCleaner import SmoothingFrameCleaner
                frame_cleaners.append(SmoothingFrameCleaner())
            if session.get("pr_gray"):
                from library.frame_cleaners.OpenCVGrayFrameCleaner import OpenCVGrayFrameCleaner
                frame_cleaners.append(OpenCVGrayFrameCleaner())

            extractor = OpenCVBufferedFrameExtractor(
                path=video_path,
                config={"resize": _resize, "stride": _stride, "max_frames": _maxf},
            )

            if mode == "Singolo oggetto":
                from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
                signal_extractor = OpenCVBufferedSignalExtractor(
                    tracker_type=_tracker, start_box=roi_box,
                )
                signal_cleaners = []
                if _mavg:
                    from library.signal_cleaners.MovingAverageCleaner import MovingAverageCleaner
                    signal_cleaners.append(MovingAverageCleaner(window_size=_window))

                _analyzer_map = {
                    "Posizione verticale":   ("library.analyzers.VerticalPositionAnalyzer",    "VerticalPositionAnalyzer"),
                    "Frequenza verticale":   ("library.analyzers.VerticalFrequencyAnalyzer",   "VerticalFrequencyAnalyzer"),
                    "Posizione orizzontale": ("library.analyzers.HoriziontalPositionAnalyzer", "HorizontalPositionAnalyzer"),
                    "Velocità verticale":    ("library.analyzers.VerticalVelocityAnalyzer",    "VerticalVelocityAnalyzer"),
                    "Velocità orizzontale":  ("library.analyzers.HorizontalVelocityAnalyzer",  "HorizontalVelocityAnalyzer"),
                    "Frequenza orizzontale": ("library.analyzers.HorizontalFrequencyAnalyzer", "HorizontalFrequencyAnalyzer"),
                }
                analyzers, skipped = [], []
                for label in (session.get("pr_analyzers") or ["Posizione verticale"]):
                    mod_path, cls_name = _analyzer_map[label]
                    try:
                        mod = importlib.import_module(mod_path)
                        analyzers.append(getattr(mod, cls_name)())
                    except Exception as exc:
                        skipped.append(f"{label} ({exc})")
                if skipped:
                    st.warning("Analizzatori non disponibili: " + ", ".join(skipped))
                if not analyzers:
                    st.error("Nessun analizzatore disponibile. Seleziona 'Posizione verticale'.")
                    st.stop()

            else:
                from library.signal_extractors.OpenCVMultiObjectSignalExtractor import OpenCVMultiObjectSignalExtractor
                from library.analyzers.MultiObjectBarrierCountingAnalyzer import MultiObjectBarrierCountingAnalyzer
                signal_extractor = OpenCVMultiObjectSignalExtractor(
                    tracker_type=_tracker,
                    start_box=roi_box,
                    max_objects=session.get("pr_max_obj") or 3,
                    similarity_threshold=session.get("pr_sim") or 0.6,
                    config={"show": False},
                )
                signal_cleaners = []
                analyzers = [MultiObjectBarrierCountingAnalyzer(barriers=barriers)]

            context = PipelineContext(
                frame_extractor=extractor, signal_extractor=signal_extractor,
                analyzers=analyzers, frame_cleaners=frame_cleaners,
                signal_cleaners=signal_cleaners,
            )

        except Exception as exc:
            st.error(f"Errore nella costruzione del contesto: {exc}")
            st.stop()

        with st.spinner("Analisi in corso…"):
            try:
                results = run_sync(context)
                session.put(session.PIPELINE_RESULTS, results)
                st.success(f"Analisi completata — {len(results)} risultato/i.")
            except Exception as exc:
                st.error(f"Pipeline fallita: {exc}")
                st.stop()

    results = session.get(session.PIPELINE_RESULTS)
    if results:
        st.divider()
        render_results(results)
