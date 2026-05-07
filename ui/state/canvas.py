"""
Streamlit session-state helpers for the pipeline canvas.

The canvas keeps only UI concerns here: selected stage, default layout and
viewport defaults. Business decisions still come from the pipeline config and
core runtime snapshots.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

import streamlit as st

CANVAS_SELECTED_STAGE = "sef_selected_stage"
CANVAS_SELECTED_STAGE_WIDGET = "sef_selected_stage_widget"
CANVAS_LAST_QUERY_STAGE = "sef_last_query_stage"
CANVAS_LAYOUT = "sef_canvas_layout"
CANVAS_VIEWPORT = "sef_canvas_viewport"
CANVAS_LAYOUT_QUERY_PARAM = "canvas_layout"
CANVAS_VIEWPORT_QUERY_PARAM = "canvas_viewport"
DEFAULT_PAN_X = -20.0
DEFAULT_PAN_Y = 0.0
DEFAULT_ZOOM = 0.92

DEFAULT_STAGE_LAYOUT: dict[str, tuple[int, int]] = {
    "frame_extractor": (120, 180),
    "frame_processors": (430, 180),
    "signal_extractor": (770, 180),
    "signal_cleaners": (1090, 180),
    "analyzers": (1420, 180),
    "visualizers": (1740, 180),
}


@dataclass(frozen=True, slots=True)
class CanvasViewport:
    """Initial viewport values consumed by the HTML canvas renderer."""

    pan_x: float = DEFAULT_PAN_X
    pan_y: float = DEFAULT_PAN_Y
    zoom: float = DEFAULT_ZOOM


def ensure_canvas_state() -> None:
    """Populate canvas-specific session defaults once per Streamlit session."""
    initial_layout = _layout_from_query() or {
        stage: {"x": x, "y": y} for stage, (x, y) in DEFAULT_STAGE_LAYOUT.items()
    }
    initial_viewport = _viewport_from_query() or {
        "pan_x": DEFAULT_PAN_X,
        "pan_y": DEFAULT_PAN_Y,
        "zoom": DEFAULT_ZOOM,
    }
    st.session_state.setdefault(CANVAS_SELECTED_STAGE, "frame_extractor")
    st.session_state.setdefault(CANVAS_LAST_QUERY_STAGE, None)
    st.session_state.setdefault(CANVAS_SELECTED_STAGE_WIDGET, "frame_extractor")
    st.session_state.setdefault(CANVAS_LAYOUT, initial_layout)
    st.session_state.setdefault(CANVAS_VIEWPORT, initial_viewport)


def sync_layout_from_query() -> None:
    """Refresh the in-memory layout when the browser URL carries a saved layout."""
    query_layout = _layout_from_query()
    if query_layout is not None:
        st.session_state[CANVAS_LAYOUT] = query_layout
    query_viewport = _viewport_from_query()
    if query_viewport is not None:
        st.session_state[CANVAS_VIEWPORT] = query_viewport


def selected_stage() -> str:
    """Return the currently selected stage in the visual composer."""
    return str(st.session_state.get(CANVAS_SELECTED_STAGE, "frame_extractor"))


def set_selected_stage(stage_key: str) -> None:
    """Update the selected stage used by the stage editor."""
    st.session_state[CANVAS_SELECTED_STAGE] = stage_key


def last_query_stage() -> str | None:
    """Return the last stage value that was applied from the URL query string."""
    return st.session_state.get(CANVAS_LAST_QUERY_STAGE)


def set_last_query_stage(stage_key: str | None) -> None:
    """Remember the last applied stage query value to avoid stale overrides."""
    st.session_state[CANVAS_LAST_QUERY_STAGE] = stage_key


def layout() -> dict[str, dict[str, int]]:
    """Return the last known node layout, or the default one."""
    ensure_canvas_state()
    return dict(st.session_state[CANVAS_LAYOUT])


def viewport() -> CanvasViewport:
    """Return initial viewport information for the embedded canvas."""
    ensure_canvas_state()
    raw = st.session_state[CANVAS_VIEWPORT]
    return CanvasViewport(
        pan_x=float(raw.get("pan_x", DEFAULT_PAN_X)),
        pan_y=float(raw.get("pan_y", DEFAULT_PAN_Y)),
        zoom=float(raw.get("zoom", DEFAULT_ZOOM)),
    )


def _layout_from_query() -> dict[str, dict[str, int]] | None:
    encoded = st.query_params.get(CANVAS_LAYOUT_QUERY_PARAM)
    if not encoded:
        return None
    try:
        raw = json.loads(base64.urlsafe_b64decode(_pad_base64(str(encoded))).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    layout: dict[str, dict[str, int]] = {}
    for stage_key, position in raw.items():
        if not isinstance(stage_key, str) or not isinstance(position, dict):
            continue
        x = position.get("x")
        y = position.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        layout[stage_key] = {"x": int(x), "y": int(y)}

    return layout or None


def _viewport_from_query() -> dict[str, float] | None:
    encoded = st.query_params.get(CANVAS_VIEWPORT_QUERY_PARAM)
    if not encoded:
        return None
    try:
        raw = json.loads(base64.urlsafe_b64decode(_pad_base64(str(encoded))).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    pan_x = raw.get("pan_x")
    pan_y = raw.get("pan_y")
    zoom = raw.get("zoom")
    if not isinstance(pan_x, (int, float)) or not isinstance(pan_y, (int, float)) or not isinstance(zoom, (int, float)):
        return None

    return {
        "pan_x": float(pan_x),
        "pan_y": float(pan_y),
        "zoom": float(zoom),
    }


def _pad_base64(value: str) -> str:
    missing = len(value) % 4
    if missing == 0:
        return value
    return value + ("=" * (4 - missing))
