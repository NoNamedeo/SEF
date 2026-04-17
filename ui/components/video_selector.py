"""
Video source selector component.

Renders a radio button (Demo / Upload), a player, and video metadata.
Returns (video_path, first_frame_bgr, metadata_dict) or (None, None, None).
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

# ── project-root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEMO_DIR = _ROOT / "videos"
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "sef_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


# ── Internal helpers (cached) ─────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _demo_paths() -> list[str]:
    if not _DEMO_DIR.exists():
        return []
    return sorted(str(p) for p in _DEMO_DIR.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_EXTS)


@st.cache_data(show_spinner="Caricamento primo frame…")
def _load_first_frame(video_path: str) -> tuple[np.ndarray, dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Impossibile leggere il primo frame.")
    meta: dict[str, Any] = {
        "fps": fps,
        "total_frames": total,
        "width": width,
        "height": height,
        "duration_s": (total / fps) if fps > 0 else None,
    }
    return frame, meta


def _save_upload(file) -> Path:
    suffix = Path(file.name).suffix or ".mp4"
    digest = hashlib.sha256(file.getbuffer()).hexdigest()[:16]
    path = _UPLOAD_DIR / f"{digest}{suffix}"
    if not path.exists():
        path.write_bytes(file.getbuffer())
    return path


# ── Public component ──────────────────────────────────────────────────────────


def render_video_selector() -> tuple[str | None, np.ndarray | None, dict | None]:
    """
    Render the video source selection widget.

    Returns
    -------
    (video_path, first_frame_bgr, metadata)  or  (None, None, None)
    """
    source = st.radio("Sorgente", ["Demo", "Upload"], horizontal=True, key="vs_source")

    video_path: str | None = None
    video_bytes: bytes | None = None

    if source == "Demo":
        demos = _demo_paths()
        if not demos:
            st.warning("Nessun video trovato in `videos/`.")
            return None, None, None
        chosen = st.selectbox(
            "Video demo",
            demos,
            format_func=lambda p: Path(p).name,
            key="vs_demo_sel",
        )
        video_path = chosen
        video_bytes = Path(chosen).read_bytes()

    else:  # Upload
        uploaded = st.file_uploader("Carica video", type=["mp4", "mov", "avi", "mkv"], key="vs_upload")
        if uploaded is None:
            st.info("Carica un file video per continuare.")
            return None, None, None
        video_path = str(_save_upload(uploaded))
        video_bytes = uploaded.getvalue()

    # ── Player ────────────────────────────────────────────────────────────────
    if video_bytes:
        with st.expander("Player video", expanded=False):
            st.video(video_bytes)

    # ── Load first frame + metadata ───────────────────────────────────────────
    try:
        frame, meta = _load_first_frame(video_path)
    except Exception as exc:
        st.error(f"Errore: {exc}")
        return None, None, None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risoluzione", f"{meta['width']}×{meta['height']}")
    c2.metric("FPS", f"{meta['fps']:.1f}")
    c3.metric("Frame totali", meta["total_frames"])
    if meta["duration_s"]:
        c4.metric("Durata", f"{meta['duration_s']:.1f}s")

    return video_path, frame, meta
