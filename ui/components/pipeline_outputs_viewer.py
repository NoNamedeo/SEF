"""Composite Streamlit viewer for pipeline outputs."""

from __future__ import annotations

import streamlit as st

from ui.components.rendering_utils import (
    materialize_video_artifact,
    render_safe_metadata,
    render_video_download,
)
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

    tab_analysis, tab_artifacts, tab_videos, tab_metadata = st.tabs(
        [
            f"Analysis ({len(view.analysis_results)})",
            f"Artifacts ({len(view.visualizer_outputs)})",
            f"Videos ({len(view.reconstructed_videos)})",
            "Run metadata",
        ]
    )

    with tab_analysis:
        if view.analysis_results:
            render_results(view.analysis_results)
        else:
            st.info("Nessun risultato analitico disponibile.")

    with tab_artifacts:
        if view.visualizer_outputs:
            selected_index = _select_artifact_index(
                [item.artifact.title or f"Artifact {index + 1}" for index, item in enumerate(view.visualizer_outputs)],
                key="sef_visualizer_artifact_selector",
            )
            item = view.visualizer_outputs[selected_index]
            st.caption(item.source)
            render_visual_artifacts(
                (item.artifact,),
                key_prefix=f"visualizer_output_{selected_index}",
            )
        else:
            st.info("Nessun artifact visuale disponibile.")

    with tab_videos:
        if view.reconstructed_videos:
            selected_index = _select_artifact_index(
                [video.title for video in view.reconstructed_videos],
                key="sef_reconstructed_video_selector",
            )
            _render_reconstructed_video(view.reconstructed_videos[selected_index], selected_index)
        else:
            st.info("Nessun video ricostruito disponibile.")

    with tab_metadata:
        render_safe_metadata("Run metadata", view.metadata, expanded=True)


def _select_artifact_index(labels: list[str], *, key: str) -> int:
    if len(labels) == 1:
        return 0
    selected_label = st.selectbox("Elemento", labels, index=0, key=key)
    return labels.index(selected_label)


def _render_reconstructed_video(video: ReconstructedVideoOutput, index: int) -> None:
    with st.container(border=True):
        st.markdown(f"**{video.title}**")
        st.caption(video.source)
        video_path = materialize_video_artifact(video.artifact)
        st.video(str(video_path), format=video.artifact.mime_type)
        render_video_download(
            video.artifact,
            key=f"reconstructed_video_{video.artifact_id}_{index}",
            label="Download video",
        )
        if video.metadata:
            render_safe_metadata("Video metadata", video.metadata, expanded=False)
