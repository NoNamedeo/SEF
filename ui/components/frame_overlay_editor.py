"""Streamlit wrapper for the frame overlay shape editor.

This helper exposes a tiny custom Streamlit component that renders the first
frame and lets the user draw a rectangle or a line directly on top of it.
The component returns small JSON events so the caller can decide how to
persist or clear the selection in session state.
"""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent / "frame_overlay_editor_component"
_OVERLAY_EDITOR = components.declare_component(
    "frame_overlay_editor",
    path=str(_COMPONENT_DIR),
)


def _frame_signature(frame_bgr: np.ndarray) -> str:
    """Return a stable fingerprint for the frame content.

    The signature is intentionally cheap to compute because it is used only to
    cache the encoded preview inside Streamlit session state.
    """
    hasher = hashlib.sha1()
    hasher.update(str(frame_bgr.shape).encode("utf-8"))

    height, width = frame_bgr.shape[:2]
    sample_points = [
        (0, 0),
        (max(0, width // 2 - 4), max(0, height // 2 - 4)),
        (max(0, width - 8), max(0, height - 8)),
    ]
    for x, y in sample_points:
        patch = frame_bgr[y : min(height, y + 8), x : min(width, x + 8)]
        hasher.update(patch.tobytes())
    return hasher.hexdigest()[:16]


def _frame_to_data_url(frame_bgr: np.ndarray) -> str:
    """Encode a BGR frame as a PNG data URL for the browser component."""
    success, buffer = cv2.imencode(".png", frame_bgr)
    if not success:
        raise RuntimeError("Impossibile codificare il frame di riferimento per l'editor overlay.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _cached_frame_data_url(frame_bgr: np.ndarray, cache_key: str) -> str:
    """Reuse the data URL while the underlying frame stays unchanged."""
    state_key = f"{cache_key}_image_data_url"
    if state_key not in st.session_state:
        st.session_state[state_key] = _frame_to_data_url(frame_bgr)
    return st.session_state[state_key]


def render_frame_overlay_editor(
    frame_bgr: np.ndarray,
    *,
    mode: Literal["rect", "line"],
    current_shape: dict[str, Any] | None = None,
    confirmed_shapes: list[dict[str, Any]] | None = None,
    current_label: str = "",
    instruction: str = "",
    stroke_color: str = "#40d67c",
    fill_color: str = "rgba(64, 214, 124, 0.16)",
    disabled: bool = False,
    key: str = "frame_overlay_editor",
) -> dict[str, Any] | None:
    """Render the overlay editor and return the latest interaction event.

    Parameters
    ----------
    frame_bgr:
        Frame already prepared in the coordinate space the caller wants to use.
    mode:
        ``"rect"`` for ROI selection, ``"line"`` for barriers.
    current_shape:
        Shape currently persisted in session state, if any.
    confirmed_shapes:
        Previously confirmed shapes to keep visible on the overlay.
    current_label / instruction:
        Human-readable labels shown inside the component.
    stroke_color / fill_color:
        CSS colors used for the active draft shape.
    disabled:
        When true the overlay is rendered read-only.
    key:
        Stable Streamlit key for this editor instance.
    """
    if frame_bgr.ndim != 3:
        raise ValueError("Il frame deve essere un array BGR a tre canali.")

    cache_key = f"{key}_{_frame_signature(frame_bgr)}"
    image_data_url = _cached_frame_data_url(frame_bgr, cache_key)

    event = _OVERLAY_EDITOR(
        image_data_url=image_data_url,
        mode=mode,
        current_shape=current_shape,
        confirmed_shapes=confirmed_shapes or [],
        current_label=current_label,
        instruction=instruction,
        stroke_color=stroke_color,
        fill_color=fill_color,
        reset_token=cache_key,
        disabled=disabled,
        key=key,
    )

    if isinstance(event, dict):
        return event
    return None
