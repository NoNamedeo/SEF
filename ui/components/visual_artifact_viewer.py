"""Streamlit renderer for pipeline visual artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import tempfile

import streamlit as st

from library.core.visualization.VisualArtifact import (
    ImageArtifact,
    JsonArtifact,
    TableArtifact,
    TextArtifact,
    VideoArtifact,
    VisualArtifact,
)


def render_visual_artifacts(
    artifacts: Sequence[VisualArtifact],
    *,
    show_metadata: bool = True,
    show_title: bool = True,
    key_prefix: str = "artifact",
) -> None:
    """Render a sequence of visual artifacts in Streamlit."""
    if not artifacts:
        st.info("Nessun artifact visuale disponibile.")
        return

    for index, artifact in enumerate(artifacts):
        title = artifact.title or f"{artifact.kind.title()} artifact {index + 1}"
        with st.container(border=True):
            if show_title:
                st.markdown(f"**{title}**")
            if artifact.description:
                st.caption(artifact.description)
            _render_artifact_body(artifact, key=f"{key_prefix}_{index}")
            if show_metadata and artifact.metadata:
                with st.expander("Artifact metadata", expanded=False):
                    st.json(dict(artifact.metadata))


def _render_artifact_body(artifact: VisualArtifact, *, key: str) -> None:
    if isinstance(artifact, ImageArtifact):
        st.image(artifact.data)
        return
    if isinstance(artifact, VideoArtifact):
        video_path = _materialize_video_artifact(artifact)
        st.video(str(video_path), format=artifact.mime_type)
        extension = ".mp4" if artifact.mime_type == "video/mp4" else ".bin"
        st.download_button(
            "Download artifact",
            data=artifact.data,
            file_name=f"{artifact.artifact_id}{extension}",
            mime=artifact.mime_type,
            key=f"{key}_download",
            width="stretch",
        )
        return
    if isinstance(artifact, TableArtifact):
        st.dataframe(list(artifact.rows), width="stretch")
        return
    if isinstance(artifact, JsonArtifact):
        st.json(dict(artifact.payload))
        return
    if isinstance(artifact, TextArtifact):
        if artifact.content_type == "text/plain":
            st.code(artifact.content)
        else:
            st.markdown(artifact.content)
        return
    st.code(repr(artifact))


def _materialize_video_artifact(artifact: VideoArtifact) -> Path:
    extension = ".mp4" if artifact.mime_type == "video/mp4" else ".bin"
    artifact_dir = Path(tempfile.gettempdir()) / "sef_streamlit_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{artifact.artifact_id}{extension}"
    if not artifact_path.exists() or artifact_path.stat().st_size != len(artifact.data):
        artifact_path.write_bytes(artifact.data)
    return artifact_path
