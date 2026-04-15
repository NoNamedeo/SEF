"""
Interactive ROI selector component.

The user draws a rectangle on the first frame with the mouse.
Returns (x, y, w, h) in the *resized* frame's coordinate space,
or None if nothing has been drawn yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Apply compatibility patch BEFORE importing streamlit_drawable_canvas.
from ui.components import _canvas_compat  # noqa: F401, E402

# Maximum canvas width in the Streamlit layout (px).
_CANVAS_MAX_W = 720


def render_roi_selector(
    frame_bgr: np.ndarray,
    resize: tuple[int, int] | None = None,
    key: str = "roi_canvas",
) -> tuple[int, int, int, int] | None:
    """
    Render an interactive ROI-drawing canvas.

    Parameters
    ----------
    frame_bgr : np.ndarray
        First video frame in BGR (as returned by cv2).
    resize : (width, height) | None
        If provided the frame is resized to this resolution before display;
        returned coordinates are in *this* space (matching the pipeline config).
    key : str
        Streamlit widget key — must be unique per page.

    Returns
    -------
    (x, y, w, h) in the (possibly resized) frame's coordinate space,
    or None if no rectangle has been drawn.
    """
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.error("Installa `streamlit-drawable-canvas` per usare il selettore ROI interattivo.")
        return None

    # ── Prepare the background image ──────────────────────────────────────────
    if resize is not None:
        target = cv2.resize(frame_bgr, resize)
    else:
        target = frame_bgr

    h_ref, w_ref = target.shape[:2]

    # Scale down to fit the UI column (never upscale).
    ui_scale = min(1.0, _CANVAS_MAX_W / w_ref)
    ui_w = int(w_ref * ui_scale)
    ui_h = int(h_ref * ui_scale)

    display = cv2.resize(target, (ui_w, ui_h)) if ui_scale < 1.0 else target
    pil_bg  = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

    st.caption("🖱 Disegna un rettangolo attorno all'oggetto da tracciare.")

    canvas_result = st_canvas(
        fill_color   = "rgba(64, 214, 124, 0.15)",
        stroke_width = 2,
        stroke_color = "#40d67c",
        background_image = pil_bg,
        update_streamlit = True,
        height       = ui_h,
        width        = ui_w,
        drawing_mode = "rect",
        key          = key,
    )

    # ── Parse canvas output ───────────────────────────────────────────────────
    if (
        canvas_result.json_data is None
        or not canvas_result.json_data.get("objects")
    ):
        return None

    obj = canvas_result.json_data["objects"][-1]

    # Canvas coords → reference-frame coords
    x = int(obj.get("left",   0) / ui_scale)
    y = int(obj.get("top",    0) / ui_scale)
    w = int(obj.get("width",  1) / ui_scale)
    h = int(obj.get("height", 1) / ui_scale)

    # Normalise negative dimensions (drawn right→left or bottom→top).
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h

    # Clamp to frame bounds.
    x = max(0, min(x, w_ref - 1))
    y = max(0, min(y, h_ref - 1))
    w = max(1, min(w, w_ref - x))
    h = max(1, min(h, h_ref - y))

    return x, y, w, h
