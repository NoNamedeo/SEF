"""Interactive ROI selector with a direct overlay on the first frame.

The user draws the ROI on top of the frame itself. The selection is stored in
session state as ``(x, y, w, h)`` in the same coordinate space used by the
pipeline configuration.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from ui.components.frame_overlay_editor import render_frame_overlay_editor


def _to_rect(selection: dict[str, Any]) -> tuple[int, int, int, int]:
    """Convert a component payload into an integer rectangle."""
    x = int(round(float(selection.get("x", 0))))
    y = int(round(float(selection.get("y", 0))))
    w = int(round(float(selection.get("w", 1))))
    h = int(round(float(selection.get("h", 1))))
    return x, y, w, h


def _clamp_rect(rect: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp a rectangle to the bounds of the current frame."""
    x, y, w, h = rect
    max_x = max(0, width - 1)
    max_y = max(0, height - 1)
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def _current_roi(state_key: str) -> tuple[int, int, int, int] | None:
    value = st.session_state.get(state_key)
    if value is None:
        return None
    x, y, w, h = value
    return int(x), int(y), int(w), int(h)


def _is_new_event(event: dict[str, Any], state_key: str) -> bool:
    """Return True only for unprocessed component events."""
    event_id = event.get("event_id")
    if not event_id:
        return False
    processed_key = f"{state_key}_last_event_id"
    if st.session_state.get(processed_key) == event_id:
        return False
    st.session_state[processed_key] = event_id
    return True


def render_roi_selector(
    frame_bgr: np.ndarray,
    resize: tuple[int, int] | None = None,
    key: str = "roi_canvas",
) -> tuple[int, int, int, int] | None:
    """Render the ROI selector and return the confirmed ROI, if any."""
    target = cv2.resize(frame_bgr, resize) if resize is not None else frame_bgr
    height, width = target.shape[:2]

    state_key = f"{key}_roi_box"
    current_roi = _current_roi(state_key)
    current_shape = None if current_roi is None else {
        "x": current_roi[0],
        "y": current_roi[1],
        "w": current_roi[2],
        "h": current_roi[3],
    }

    st.caption("Trascina il rettangolo direttamente sul frame e rilascia per confermare.")

    event = render_frame_overlay_editor(
        target,
        mode="rect",
        current_shape=current_shape,
        confirmed_shapes=[],
        current_label="ROI",
        instruction="Trascina il rettangolo sopra il frame. Rilascia per confermare. Usa Azzera per rimuoverla.",
        stroke_color="#40d67c",
        fill_color="rgba(64, 214, 124, 0.18)",
        key=f"{key}_editor",
    )

    if isinstance(event, dict) and _is_new_event(event, state_key):
        action = event.get("action")
        if action == "draw":
            shape = event.get("shape")
            if isinstance(shape, dict):
                st.session_state[state_key] = _clamp_rect(_to_rect(shape), width, height)
        elif action == "clear":
            if state_key in st.session_state:
                del st.session_state[state_key]

    return _current_roi(state_key)
