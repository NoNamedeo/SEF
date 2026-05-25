"""Streamlit adapter for realtime frame sinks."""

from __future__ import annotations

import streamlit as st

from ui.services.realtime_mjpeg_server import mjpeg_stream_url
from ui.services.realtime_preview_service import snapshot_for_id


def render_realtime_preview(sink_id: str, *, title: str = "Realtime preview") -> None:
    """Render a browser-native MJPEG preview for a realtime sink."""
    st.markdown(f"### {title}")
    st.caption(f"Sink: `{sink_id}`")
    snapshot = snapshot_for_id(sink_id)
    frame = snapshot.frame

    status_cols = st.columns(4)
    status_cols[0].metric("Frame", "-" if frame is None or frame.frame_index is None else frame.frame_index)
    status_cols[1].metric("Version", snapshot.version)
    status_cols[2].metric("Frames", snapshot.published_frames)
    status_cols[3].metric("Stage", snapshot.last_stage or ("live" if snapshot.active else "idle"))

    stream_url = mjpeg_stream_url(sink_id)
    st.markdown(
        (
            '<div style="width:100%; background:#111; border:1px solid #2f343b; border-radius:8px; overflow:hidden;">'
            f'<img src="{stream_url}" style="display:block; width:100%; height:auto;" />'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
