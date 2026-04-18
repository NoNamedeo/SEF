"""Presentation component for analysis results."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from ui.components.visual_artifact_viewer import render_visual_artifacts
from ui.models.pipeline_outputs import AnalysisResultOutput


def render_results(results: Sequence[AnalysisResultOutput]) -> None:
    """Render analysis outputs using the explicit UI view model."""
    if not results:
        st.warning("Nessun risultato disponibile.")
        return

    for result in results:
        with st.container(border=True):
            st.markdown(f"**{result.title}**")
            st.caption(result.type_name)

            if result.summary:
                _render_summary_metrics(result.summary)

            if result.preview_artifacts:
                st.markdown("Anteprima UI")
                render_visual_artifacts(
                    result.preview_artifacts,
                    show_metadata=False,
                    show_title=False,
                    key_prefix=result.result_id,
                )

            if result.detail_rows:
                with st.expander("Dettaglio analitico", expanded=False):
                    st.dataframe(list(result.detail_rows), width="stretch")

            if result.metadata:
                with st.expander("Metadati analisi", expanded=False):
                    st.json(dict(result.metadata))

            if not result.preview_artifacts and not result.summary and not result.detail_rows:
                st.code(str(result.data))


def _render_summary_metrics(summary: dict[str, object]) -> None:
    items = list(summary.items())
    if not items:
        return
    column_count = min(max(len(items), 1), 4)
    columns = st.columns(column_count)
    for index, (label, value) in enumerate(items):
        columns[index % column_count].metric(str(label), value)
