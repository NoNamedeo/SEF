"""Presentation component for analysis results."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from ui.components.rendering_utils import render_safe_metadata
from ui.components.visual_artifact_viewer import render_visual_artifacts
from ui.models.pipeline_outputs import AnalysisResultOutput


def render_results(results: Sequence[AnalysisResultOutput]) -> None:
    """Render analysis outputs using the explicit UI view model."""
    if not results:
        st.warning("Nessun risultato disponibile.")
        return

    selected_result = _select_result(results)
    if selected_result is None:
        return

    _render_result(selected_result)


def _select_result(results: Sequence[AnalysisResultOutput]) -> AnalysisResultOutput | None:
    if len(results) == 1:
        return results[0]

    labels = [f"{index + 1}. {result.title} ({result.type_name})" for index, result in enumerate(results)]
    selected_label = st.selectbox(
        "Risultato analitico",
        labels,
        index=0,
        key="sef_analysis_result_selector",
    )
    selected_index = labels.index(selected_label)
    return results[selected_index]


def _render_result(result: AnalysisResultOutput) -> None:
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
                    st.dataframe(list(result.detail_rows[:250]), width="stretch")
                    if len(result.detail_rows) > 250:
                        st.caption(f"Mostrate 250 righe su {len(result.detail_rows)} per stabilita UI.")

            if result.metadata:
                render_safe_metadata("Metadati analisi", result.metadata, expanded=False)

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
