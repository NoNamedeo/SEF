"""
Results viewer component.

Dispatches on IData concrete type and renders the appropriate chart/table.
New data types only require adding a new elif branch here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_results(results: list) -> None:
    """
    Render a list of IData objects returned by PipelineOrchestrator.run().

    Supports
    --------
    TwoDimGraphData  → line/scatter chart  (MatplotlibFunctionVisualizer)
    CategoryData     → bar chart           (MatplotlibHistogramVisualizer)
    TrajectoryData   → trajectory chart    (MatplotlibTrajectoryVisualizer)  [if available]
    anything else    → raw string dump
    """
    if not results:
        st.warning("Nessun risultato disponibile.")
        return

    from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
    from library.core.artifacts.CategoryData import CategoryData

    for i, data in enumerate(results):
        st.divider()

        if isinstance(data, TwoDimGraphData):
            _render_two_dim(data, idx=i)

        elif isinstance(data, CategoryData):
            _render_category(data, idx=i)

        else:
            # Try trajectory if available
            try:
                from library.core.artifacts.TrajectoryData import TrajectoryData
                if isinstance(data, TrajectoryData):
                    _render_trajectory(data, idx=i)
                    continue
            except ImportError:
                pass
            # Fallback
            st.warning(f"Tipo `{type(data).__name__}` non ha un renderer dedicato.")
            st.code(str(data))


# ── Sub-renderers ─────────────────────────────────────────────────────────────

def _render_two_dim(data, idx: int = 0) -> None:
    from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

    st.subheader(data.title or f"Risultato {idx + 1}")

    viz = MatplotlibFunctionVisualizer(config={"show": False, "show_scatter": True})
    fig, _ = viz.visualize(data)
    st.pyplot(fig, use_container_width=True)

    # Summary metrics
    if data.x and data.y:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Campioni",  len(data.x))
        c2.metric(f"Min {data.y_label}", f"{min(data.y):.2f}")
        c3.metric(f"Max {data.y_label}", f"{max(data.y):.2f}")
        mean_y = sum(data.y) / len(data.y)
        c4.metric(f"Media {data.y_label}", f"{mean_y:.2f}")

    # Collapsible metadata
    if data.metadata:
        with st.expander("Metadati analisi"):
            st.json(data.metadata)


def _render_category(data, idx: int = 0) -> None:
    from library.visualizers.MatplotlibHistogramVisualizer import MatplotlibHistogramVisualizer

    st.subheader(f"Conteggio attraversamenti barriere")

    viz = MatplotlibHistogramVisualizer(config={"show": False})
    fig, _ = viz.visualize(data)
    st.pyplot(fig, use_container_width=True)

    # Metrics per category
    if data.categories:
        cols = st.columns(min(len(data.categories), 4))
        for i, cat in enumerate(data.categories):
            cols[i % len(cols)].metric(cat, data.category_counts.get(cat, 0))

    # Metadata
    if data.metadata:
        with st.expander("Metadati analisi"):
            st.json(data.metadata)

    # Track details
    if data.track_categories:
        with st.expander(f"Dettaglio per oggetto tracciato ({len(data.track_categories)} oggetti)"):
            rows = [
                {"track_id": tid, "barriere_attraversate": ", ".join(cats)}
                for tid, cats in sorted(data.track_categories.items())
            ]
            st.dataframe(rows, use_container_width=True)


def _render_trajectory(data, idx: int = 0) -> None:
    try:
        from library.visualizers.MatplotlibTrajectoryVisualizer import MatplotlibTrajectoryVisualizer
        st.subheader("Traiettoria")
        viz = MatplotlibTrajectoryVisualizer(config={"show": False})
        fig, _ = viz.visualize(data)
        st.pyplot(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Visualizzazione traiettoria non disponibile: {exc}")
        st.code(str(data))
