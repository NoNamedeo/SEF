"""Composite Streamlit viewer for pipeline outputs."""

from __future__ import annotations

import streamlit as st

from library.core.visualization.PipelineOutputs import PipelineOutputs
from ui.components.results_viewer import render_results
from ui.components.visual_artifact_viewer import render_visual_artifacts
from ui.models.pipeline_outputs import ReconstructedVideoOutput
from ui.services.pipeline_outputs_service import build_execution_results_view


def render_pipeline_outputs(outputs: PipelineOutputs, *, title: str | None = None) -> None:
    """Render analysis data, configured visualizer outputs, and reconstructed videos."""
    if title:
        st.markdown(f"### {title}")

    view = build_execution_results_view(outputs)

    if view.warnings:
        for warning in view.warnings:
            st.warning(warning)

    st.markdown(f"**Analysis results ({len(view.analysis_results)})**")
    if view.analysis_results:
        render_results(view.analysis_results)
    else:
        st.info("Nessun risultato analitico disponibile.")

    st.divider()
    st.markdown(f"**Configured visualizer outputs ({len(view.visualizer_outputs)})**")
    if view.visualizer_outputs:
        for index, item in enumerate(view.visualizer_outputs):
            st.caption(item.source)
            render_visual_artifacts(
                (item.artifact,),
                key_prefix=f"visualizer_output_{index}",
            )
    else:
        st.info("Nessun artifact visuale disponibile.")

    st.divider()
    st.markdown(f"**Reconstructed videos ({len(view.reconstructed_videos)})**")
    if view.reconstructed_videos:
        for index, video in enumerate(view.reconstructed_videos):
            _render_reconstructed_video(video, index)
    else:
        st.info("Nessun video ricostruito disponibile.")

    with st.expander("Run metadata", expanded=False):
        st.json(dict(view.metadata))


def _render_reconstructed_video(video: ReconstructedVideoOutput, index: int) -> None:
    with st.container(border=True):
        st.markdown(f"**{video.title}**")
        st.caption(video.source)
        st.video(video.data, format=video.mime_type)
        st.download_button(
            "Download video",
            data=video.data,
            file_name=f"reconstructed_video_{index + 1}.mp4",
            mime=video.mime_type,
            key=f"reconstructed_video_{video.artifact_id}_{index}",
            width="stretch",
        )
        if video.metadata:
            with st.expander("Video metadata", expanded=False):
                st.json(dict(video.metadata))
