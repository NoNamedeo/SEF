"""
Centralised session-state keys and accessors.

Every key used across pages lives here — no magic strings scattered around.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

# ── Keys ──────────────────────────────────────────────────────────────────────

VIDEO_PATH       = "sef_video_path"         # str
VIDEO_BYTES      = "sef_video_bytes"        # bytes
FIRST_FRAME      = "sef_first_frame"        # np.ndarray  BGR
FRAME_META       = "sef_frame_meta"         # dict  {fps, width, height, total_frames, duration_s}
ROI_BOX          = "sef_roi_box"            # tuple[int,int,int,int]  (x,y,w,h)
BARRIERS         = "sef_barriers"           # dict[str, ((x1,y1),(x2,y2))]
PIPELINE_RESULTS = "sef_pipeline_results"   # list[IData]
PIPELINE_CONFIG  = "sef_pipeline_config"    # dict  — last config used in Config Builder
LOG_RECORDS      = "sef_log_records"        # list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def put(key: str, value: Any) -> None:
    st.session_state[key] = value


def clear(key: str) -> None:
    if key in st.session_state:
        del st.session_state[key]


def clear_run_state() -> None:
    """Reset all pipeline-run-specific keys (call before a new run)."""
    for k in (PIPELINE_RESULTS, ROI_BOX, BARRIERS):
        clear(k)
