"""Composite Streamlit viewer for pipeline outputs."""

from __future__ import annotations

import streamlit as st

from library.core.visualization.PipelineOutputs import PipelineOutputs
from ui.components.results_viewer import render_results
from ui.components.visual_artifact_viewer import render_visual_artifacts


def render_pipeline_outputs(outputs: PipelineOutputs, *, title: str | None = None) -> None:
    """Render analysis results, visual artifacts and run metadata."""
    if title:
        st.markdown(f"### {title}")

    st.markdown("**Analysis results**")
    if outputs.results:
        render_results(list(outputs.results))
    else:
        st.info("Nessun risultato analitico disponibile.")

    st.divider()
    st.markdown("**Visualization artifacts**")
    render_visual_artifacts(outputs.artifacts)

    with st.expander("Run metadata", expanded=False):
        st.json(
            {
                "pipeline_id": outputs.metadata.pipeline_id,
                "generated_at": outputs.metadata.generated_at.isoformat(),
                "execution_metadata": dict(outputs.metadata.execution_metadata),
            }
        )
