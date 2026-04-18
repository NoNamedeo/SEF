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
BARRIER_SELECTION_STATE = "sef_barrier_selection_state"  # BarrierSelectionState
PIPELINE_OUTPUTS = "sef_pipeline_outputs"   # PipelineOutputs
PIPELINE_CONFIG  = "sef_pipeline_config"    # dict  — last config used in Config Builder
PIPELINE_CONFIG_EDITOR_RAW = "sef_pipeline_config_editor_raw"  # str
PIPELINE_CONFIG_EDITOR_BASELINE = "sef_pipeline_config_editor_baseline"  # str
PIPELINE_CONFIG_EDITOR_WIDGET = "sef_pipeline_config_editor_widget"  # str
LOG_RECORDS      = "sef_log_records"        # list[dict]
TRACKING_VIDEO_CACHE = "sef_tracking_video_cache"  # dict[str, dict[str, Any]]


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
    for k in (PIPELINE_OUTPUTS, ROI_BOX, BARRIERS, TRACKING_VIDEO_CACHE):
        clear(k)
