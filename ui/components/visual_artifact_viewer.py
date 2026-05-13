"""Streamlit renderer for pipeline visual artifacts."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from library.core.visualization.VisualArtifact import (
    VIDEO_ARTIFACT_TYPES,
    ImageArtifact,
    JsonArtifact,
    TableArtifact,
    TextArtifact,
    VisualArtifact,
)
from ui.components.rendering_utils import (
    MAX_IMAGE_RENDER_BYTES,
    materialize_video_artifact,
    render_safe_metadata,
    render_video_download,
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
                render_safe_metadata("Artifact metadata", artifact.metadata, expanded=False)


def _render_artifact_body(artifact: VisualArtifact, *, key: str) -> None:
    if isinstance(artifact, ImageArtifact):
        if len(artifact.data) > MAX_IMAGE_RENDER_BYTES:
            st.warning(f"Preview non renderizzata: immagine oltre {MAX_IMAGE_RENDER_BYTES} bytes.")
            st.download_button(
                "Download image",
                data=artifact.data,
                file_name=f"{artifact.artifact_id}.png",
                mime=artifact.mime_type,
                key=f"{key}_download",
                width="stretch",
            )
            return
        st.image(artifact.data)
        return
    if isinstance(artifact, VIDEO_ARTIFACT_TYPES):
        video_path = materialize_video_artifact(artifact)
        st.video(str(video_path), format=artifact.mime_type)
        render_video_download(artifact, key=f"{key}_download", label="Download artifact")
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
