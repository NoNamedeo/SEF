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
from ui.models.pipeline_outputs import (
    ExecutionResultsView,
    IntermediateFrameSnapshot,
    ReconstructedVideoOutput,
)
from ui.services.pipeline_outputs_service import build_execution_results_view


def render_pipeline_outputs(outputs: PipelineOutputs, *, title: str | None = None) -> None:
    """Render analysis data, configured visualizer outputs, reconstructed videos, and intermediate frames."""
    if title:
        st.markdown(f"### {title}")

    view = build_execution_results_view(outputs)

    if view.warnings:
        for warning in view.warnings:
            st.warning(warning)

    tab_labels = [
        f"Analysis ({len(view.analysis_results)})",
        f"Artifacts ({len(view.visualizer_outputs)})",
        f"Videos ({len(view.reconstructed_videos)})",
    ]
    if view.intermediate_frame_count > 0:
        tab_labels.append(f"Intermediate frames ({view.intermediate_frame_count})")
    tab_labels.append("Run metadata")

    tabs = st.tabs(tab_labels)

    tab_analysis, tab_artifacts, tab_videos = tabs[0], tabs[1], tabs[2]
    remaining = list(tabs[3:])

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

    if view.intermediate_frame_count > 0:
        tab_intermediate = remaining[0]
        remaining = remaining[1:]
        with tab_intermediate:
            render_intermediate_frame_comparison(view)

    tab_metadata = remaining[0] if remaining else tabs[-1]
    with tab_metadata:
        render_safe_metadata("Run metadata", view.metadata, expanded=True)


def render_intermediate_frame_comparison(view: ExecutionResultsView) -> None:
    """Render an interactive intermediate frame comparison viewer."""
    snapshots = list(view.intermediate_frame_snapshots)
    if not snapshots:
        st.info("Nessun frame intermedio disponibile.")
        return

    st.caption(
        "Confronta i frame originali con quelli processati da ogni stage. Filtra per stage o frame index e scegli la modalita di visualizzazione."
    )

    # ── Filters ──────────────────────────────────────────────────────────────
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    stages = list(view.intermediate_frame_stages)
    all_stages_label = "Tutti gli stages"
    stage_options = [all_stages_label] + stages

    selected_stage = col_filter1.selectbox(
        "Filtra per stage",
        stage_options,
        key="sef_if_stage_filter",
    )

    frame_indices = sorted(
        {s.frame_index for s in snapshots if s.frame_index is not None},
    )
    all_frames_label = "Tutti i frame"
    frame_options = [all_frames_label] + [str(idx) for idx in frame_indices]
    selected_frame_str = col_filter2.selectbox(
        "Filtra per frame index",
        frame_options,
        key="sef_if_frame_filter",
    )

    view_mode = col_filter3.radio(
        "Modalita",
        ["Single frame", "Grid (side-by-side)"],
        key="sef_if_view_mode",
        horizontal=True,
    )

    # ── Apply filters ────────────────────────────────────────────────────────
    filtered = list(snapshots)
    if selected_stage != all_stages_label:
        filtered = [s for s in filtered if s.stage_name == selected_stage]
    if selected_frame_str != all_frames_label:
        target_idx = int(selected_frame_str)
        filtered = [s for s in filtered if s.frame_index == target_idx]

    if not filtered:
        st.info("Nessun frame intermedio corrispondente ai filtri selezionati.")
        return

    st.metric("Frame mostrati", len(filtered))

    # ── Render based on view mode ────────────────────────────────────────────
    if view_mode == "Grid (side-by-side)":
        _render_intermediate_grid(filtered)
    else:
        _render_intermediate_single(filtered)


def _render_intermediate_single(snapshots: list[IntermediateFrameSnapshot]) -> None:
    """Render each intermediate frame individually."""
    index = st.selectbox(
        "Seleziona snapshot",
        range(len(snapshots)),
        format_func=lambda i: (
            f"Frame #{snapshots[i].frame_index or '?'} — "
            f"{snapshots[i].stage_name} "
            f"({'processed' if 'after' in snapshots[i].stage_name.lower() or snapshots[i].stage_name.lower() != 'original' else 'original'})"
        ),
        key="sef_if_single_selector",
    )
    snapshot = snapshots[index]
    _render_snapshot_card(snapshot, index)


def _render_intermediate_grid(snapshots: list[IntermediateFrameSnapshot]) -> None:
    """Render intermediate frames in a responsive grid layout."""
    cols_per_row = st.slider(
        "Colonne per riga",
        1,
        4,
        2,
        key="sef_if_grid_cols",
    )
    for i in range(0, len(snapshots), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx >= len(snapshots):
                break
            snapshot = snapshots[idx]
            with cols[j]:
                _render_snapshot_card(snapshot, idx, compact=True)


def _render_snapshot_card(
    snapshot: IntermediateFrameSnapshot,
    index: int,
    compact: bool = False,
) -> None:
    """Render a single intermediate frame snapshot as a card."""
    with st.container(border=True):
        frame_label = f"#{snapshot.frame_index}" if snapshot.frame_index is not None else "?"
        timestamp_str = f"{snapshot.timestamp_seconds:.3f}s" if snapshot.timestamp_seconds is not None else "-"
        if not compact:
            st.markdown(f"**{snapshot.stage_name}** — Frame {frame_label}")
            st.caption(f"Timestamp: {timestamp_str} | Color space: {snapshot.color_space}")
        else:
            st.caption(f"{snapshot.stage_name} | Frame {frame_label} | {timestamp_str}")

        st.image(
            snapshot.image_bytes,
            use_container_width=True,
        )

        # Download button for each frame
        st.download_button(
            label=f"Download PNG",
            data=snapshot.image_bytes,
            file_name=f"intermediate_{snapshot.stage_name}_f{frame_label}.png",
            mime=snapshot.mime_type,
            key=f"dl_if_{index}_{snapshot.stage_name}_{frame_label}",
            use_container_width=compact,
        )


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
