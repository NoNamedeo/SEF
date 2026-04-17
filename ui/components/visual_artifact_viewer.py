"""Streamlit renderer for pipeline visual artifacts."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from library.core.visualization.VisualArtifact import (
    ImageArtifact,
    JsonArtifact,
    TableArtifact,
    TextArtifact,
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
            _render_artifact_body(artifact)
            if show_metadata and artifact.metadata:
                with st.expander("Artifact metadata", expanded=False):
                    st.json(dict(artifact.metadata))


def _render_artifact_body(artifact: VisualArtifact) -> None:
    if isinstance(artifact, ImageArtifact):
        st.image(artifact.data)
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
