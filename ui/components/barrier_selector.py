"""
Interactive barrier-drawing component.

The user draws one line per barrier sequentially; each confirmed line is
overlaid on the background frame.  Returns a dict of
{barrier_name: ((x1, y1), (x2, y2))} in reference-frame coordinates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Apply compatibility patch BEFORE importing streamlit_drawable_canvas.
from ui.components import _canvas_compat  # noqa: F401, E402

_CANVAS_MAX_W = 720
_PALETTE = ["#FFD232", "#FF6B6B", "#6BAAFF", "#B86BFF", "#6BFFB8", "#FF9F40"]

Barrier = tuple[tuple[float, float], tuple[float, float]]


def _overlay_confirmed(
    frame_bgr: np.ndarray,
    drawn: dict[str, Barrier],
    barrier_names: list[str],
    ui_scale: float,
) -> Image.Image:
    """Return a PIL image with already-confirmed barriers drawn on top."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ui_w, ui_h = int(w * ui_scale), int(h * ui_scale)
    if ui_scale < 1.0:
        rgb = cv2.resize(rgb, (ui_w, ui_h))
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    for idx, name in enumerate(barrier_names):
        if name not in drawn:
            continue
        (x1, y1), (x2, y2) = drawn[name]
        color = _PALETTE[idx % len(_PALETTE)]
        sx1, sy1 = int(x1 * ui_scale), int(y1 * ui_scale)
        sx2, sy2 = int(x2 * ui_scale), int(y2 * ui_scale)
        draw.line([(sx1, sy1), (sx2, sy2)], fill=color, width=3)
        # label near start point
        lx = min(sx1, sx2) + 4
        ly = min(sy1, sy2) - 18
        draw.rectangle([lx - 2, ly - 2, lx + len(name) * 8, ly + 16], fill="#00000099")
        draw.text((lx, ly), name, fill=color)

    return img


def render_barrier_selector(
    frame_bgr: np.ndarray,
    barrier_names: list[str],
    resize: tuple[int, int] | None = None,
    state_key: str = "barriers",
) -> dict[str, Barrier]:
    """
    Step-by-step interactive barrier-drawing widget.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Reference frame (BGR).
    barrier_names : list[str]
        Ordered list of barrier names the user must draw.
    resize : (width, height) | None
        Optional resize applied to the reference frame (must match the
        pipeline's frame_extractor resize setting so coordinates align).
    state_key : str
        Prefix for Streamlit session-state keys (unique per page use).

    Returns
    -------
    dict  {barrier_name: ((x1, y1), (x2, y2))}  in reference-frame coords.
    """
    try:
        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.error("Installa `streamlit-drawable-canvas` per disegnare le barriere.")
        return {}

    if not barrier_names:
        st.info("Nessuna barriera richiesta.")
        return {}

    # ── Apply optional resize to the reference frame ──────────────────────────
    ref = cv2.resize(frame_bgr, resize) if resize else frame_bgr
    h_ref, w_ref = ref.shape[:2]
    ui_scale = min(1.0, _CANVAS_MAX_W / w_ref)

    # ── Session state keys ────────────────────────────────────────────────────
    k_data = f"{state_key}_data"
    k_idx  = f"{state_key}_idx"

    if k_data not in st.session_state:
        st.session_state[k_data] = {}
    if k_idx not in st.session_state:
        st.session_state[k_idx] = 0

    drawn: dict[str, Barrier] = st.session_state[k_data]
    cur_idx: int               = st.session_state[k_idx]

    # ── All barriers confirmed ────────────────────────────────────────────────
    if cur_idx >= len(barrier_names):
        final_img = _overlay_confirmed(ref, drawn, barrier_names, ui_scale)
        st.success(f"✅ Tutte le {len(barrier_names)} barriere definite.")
        col_img, col_ctrl = st.columns([2, 1])
        col_img.image(final_img, use_container_width=True)
        with col_ctrl:
            st.markdown("**Barriere confermate**")
            for name, ((x1, y1), (x2, y2)) in drawn.items():
                st.write(f"• **{name}**: ({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f})")
            if st.button("↺ Ridisegna tutto", key=f"{state_key}_reset_done"):
                st.session_state[k_data] = {}
                st.session_state[k_idx]  = 0
                st.rerun()
        return drawn

    # ── Draw current barrier ──────────────────────────────────────────────────
    cur_name  = barrier_names[cur_idx]
    cur_color = _PALETTE[cur_idx % len(_PALETTE)]

    st.info(
        f"**Barriera {cur_idx + 1} / {len(barrier_names)}: "
        f"`{cur_name}`** — clicca e trascina per disegnare la linea."
    )

    bg_img = _overlay_confirmed(ref, drawn, barrier_names, ui_scale)
    ui_w   = int(w_ref * ui_scale)
    ui_h   = int(h_ref * ui_scale)

    canvas_result = st_canvas(
        stroke_width     = 3,
        stroke_color     = cur_color,
        background_image = bg_img,
        update_streamlit = True,
        height           = ui_h,
        width            = ui_w,
        drawing_mode     = "line",
        key              = f"{state_key}_canvas_{cur_idx}",
    )

    has_line = (
        canvas_result.json_data is not None
        and canvas_result.json_data.get("objects")
    )

    col_ok, col_skip, col_reset = st.columns(3)

    if col_ok.button("✓ Conferma", disabled=not has_line, key=f"{state_key}_ok_{cur_idx}"):
        obj = canvas_result.json_data["objects"][-1]
        # st_canvas line objects expose x1/y1/x2/y2
        raw_x1 = float(obj.get("x1", obj.get("left",  0)))
        raw_y1 = float(obj.get("y1", obj.get("top",   0)))
        raw_x2 = float(obj.get("x2", raw_x1 + obj.get("width",  0)))
        raw_y2 = float(obj.get("y2", raw_y1 + obj.get("height", 0)))
        # scale back to reference-frame coordinates
        drawn[cur_name] = (
            (raw_x1 / ui_scale, raw_y1 / ui_scale),
            (raw_x2 / ui_scale, raw_y2 / ui_scale),
        )
        st.session_state[k_data] = drawn
        st.session_state[k_idx]  = cur_idx + 1
        st.rerun()

    if col_skip.button("→ Salta", key=f"{state_key}_skip_{cur_idx}"):
        st.session_state[k_idx] = cur_idx + 1
        st.rerun()

    if col_reset.button("↺ Azzera tutto", key=f"{state_key}_reset_{cur_idx}"):
        st.session_state[k_data] = {}
        st.session_state[k_idx]  = 0
        st.rerun()

    return drawn
