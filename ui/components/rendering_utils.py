"""Shared safety helpers for rendering large pipeline outputs in Streamlit."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from library.core.visualization.VisualArtifact import (
    DeferredVideoArtifact,
    VideoArtifact,
    VideoFileArtifact,
    VideoLikeArtifact,
)

MAX_METADATA_ITEMS = 24
MAX_SEQUENCE_ITEMS = 16
MAX_STRING_LENGTH = 240
MAX_JSON_DEPTH = 3
MAX_IMAGE_RENDER_BYTES = 8 * 1024 * 1024
MAX_VIDEO_DOWNLOAD_BYTES = 25 * 1024 * 1024


def render_safe_metadata(
    title: str,
    metadata: Mapping[str, Any] | None,
    *,
    expanded: bool = False,
) -> None:
    """Render metadata with truncation so large nested payloads do not crash the UI."""
    if not metadata:
        return
    with st.expander(title, expanded=expanded):
        st.json(sanitize_for_json(metadata))


def sanitize_for_json(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = MAX_JSON_DEPTH,
) -> Any:
    """Convert arbitrary Python values into a small JSON-safe structure."""
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if len(value) <= MAX_STRING_LENGTH:
            return value
        return f"{value[:MAX_STRING_LENGTH]}... ({len(value)} chars)"

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        items = list(value.items())
        limited_items = items[:MAX_METADATA_ITEMS]
        sanitized = {
            str(key): sanitize_for_json(item, depth=depth + 1, max_depth=max_depth)
            for key, item in limited_items
        }
        if len(items) > MAX_METADATA_ITEMS:
            sanitized["_truncated_items"] = len(items) - MAX_METADATA_ITEMS
        if depth >= max_depth and items:
            return {
                "_type": type(value).__name__,
                "_items": sanitized,
                "_truncated": len(items) > MAX_METADATA_ITEMS,
            }
        return sanitized

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        items = list(value[:MAX_SEQUENCE_ITEMS])
        sanitized_items = [sanitize_for_json(item, depth=depth + 1, max_depth=max_depth) for item in items]
        if len(value) > MAX_SEQUENCE_ITEMS:
            sanitized_items.append(f"... ({len(value) - MAX_SEQUENCE_ITEMS} more items)")
        if depth >= max_depth and items:
            return {
                "_type": type(value).__name__,
                "_length": len(value),
                "_preview": sanitized_items,
            }
        return sanitized_items

    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{type(value).__name__}: {len(value)} bytes>"

    return str(value)


def materialize_video_artifact(artifact: VideoLikeArtifact) -> Path:
    """Return a local file path for any supported video artifact."""
    if isinstance(artifact, VideoFileArtifact):
        return artifact.path
    if isinstance(artifact, DeferredVideoArtifact):
        return artifact.materialize(_video_artifact_dir())
    if not isinstance(artifact, VideoArtifact):
        raise TypeError(f"Unsupported video artifact type: {type(artifact).__name__}")

    artifact_dir = Path(tempfile.gettempdir()) / "sef_streamlit_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    extension = _video_extension(artifact.mime_type)
    artifact_path = artifact_dir / f"{artifact.artifact_id}{extension}"
    if not artifact_path.exists() or artifact_path.stat().st_size != len(artifact.data):
        artifact_path.write_bytes(artifact.data)
    return artifact_path


def render_video_download(artifact: VideoLikeArtifact, *, key: str, label: str) -> None:
    """Render a download button only for manageable artifact sizes."""
    artifact_path = materialize_video_artifact(artifact)
    artifact_size = artifact_path.stat().st_size
    if artifact_size > MAX_VIDEO_DOWNLOAD_BYTES:
        size_mb = artifact_size / (1024 * 1024)
        st.caption(f"Download nascosto per stabilita UI ({size_mb:.1f} MB).")
        return

    st.download_button(
        label,
        data=artifact_path.read_bytes(),
        file_name=f"{artifact.artifact_id}{_video_extension(artifact.mime_type)}",
        mime=artifact.mime_type,
        key=key,
        width="stretch",
    )


def _video_artifact_dir() -> Path:
    artifact_dir = Path(tempfile.gettempdir()) / "sef_streamlit_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _video_extension(mime_type: str) -> str:
    return ".mp4" if mime_type == "video/mp4" else ".bin"
