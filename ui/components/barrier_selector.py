"""Interactive barrier selector with a live overlay on the first frame.

Each barrier is drawn directly on top of the reference frame. The current
segment is confirmed as soon as the user releases the pointer, and the last
confirmed barrier can be undone with the component reset button.
"""
from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np
import streamlit as st

from ui.components.frame_overlay_editor import render_frame_overlay_editor

Barrier = tuple[tuple[float, float], tuple[float, float]]

_PALETTE = ["#FFD232", "#FF6B6B", "#6BAAFF", "#B86BFF", "#6BFFB8", "#FF9F40"]


def _to_barrier(selection: dict[str, Any]) -> Barrier:
    """Convert a component payload into a barrier segment."""
    x1 = float(selection.get("x1", 0))
    y1 = float(selection.get("y1", 0))
    x2 = float(selection.get("x2", 0))
    y2 = float(selection.get("y2", 0))
    return ((x1, y1), (x2, y2))


def _clamp_barrier(barrier: Barrier, width: int, height: int) -> Barrier:
    """Clamp a barrier to the frame bounds."""
    (x1, y1), (x2, y2) = barrier
    max_x = max(0, width - 1)
    max_y = max(0, height - 1)
    x1 = float(max(0, min(int(round(x1)), max_x)))
    y1 = float(max(0, min(int(round(y1)), max_y)))
    x2 = float(max(0, min(int(round(x2)), max_x)))
    y2 = float(max(0, min(int(round(y2)), max_y)))
    return ((x1, y1), (x2, y2))


def _barrier_shape(barrier: Barrier) -> dict[str, float]:
    """Return a JSON-serialisable shape for the overlay component."""
    (x1, y1), (x2, y2) = barrier
    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
    }


def _names_signature(barrier_names: list[str]) -> str:
    """Return a short hash for the current ordered list of barrier names."""
    hasher = hashlib.sha1()
    for name in barrier_names:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:12]


def _confirmed_shapes(
    drawn: dict[str, Barrier],
    barrier_names: list[str],
) -> list[dict[str, Any]]:
    """Build the confirmed-shape payload expected by the overlay component."""
    shapes: list[dict[str, Any]] = []
    for index, name in enumerate(barrier_names):
        if name not in drawn:
            continue
        shapes.append(
            {
                "label": name,
                "color": _PALETTE[index % len(_PALETTE)],
                "shape": _barrier_shape(drawn[name]),
            }
        )
    return shapes


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


def render_barrier_selector(
    frame_bgr: np.ndarray,
    barrier_names: list[str],
    resize: tuple[int, int] | None = None,
    state_key: str = "barriers",
) -> dict[str, Barrier]:
    """Render the barrier selector and return the confirmed barriers.

    Parameters
    ----------
    frame_bgr:
        First frame in BGR format.
    barrier_names:
        Ordered names that define the drawing sequence.
    resize:
        Optional working resolution used by the pipeline.
    state_key:
        Prefix for the selector's internal session-state keys.
    """
    if not barrier_names:
        st.info("Nessuna barriera richiesta.")
        return {}

    target = cv2.resize(frame_bgr, resize) if resize is not None else frame_bgr
    height, width = target.shape[:2]

    selector_sig = f"{width}x{height}_{_names_signature(barrier_names)}"
    k_data = f"{state_key}_{selector_sig}_data"
    k_idx = f"{state_key}_{selector_sig}_idx"
    k_final_reset = f"{state_key}_{selector_sig}_reset_done"

    if k_data not in st.session_state:
        st.session_state[k_data] = {}
    if k_idx not in st.session_state:
        st.session_state[k_idx] = 0

    drawn: dict[str, Barrier] = st.session_state[k_data]
    cur_idx = int(st.session_state[k_idx])
    cur_idx = max(0, min(cur_idx, len(barrier_names)))
    st.session_state[k_idx] = cur_idx

    confirmed_shapes = _confirmed_shapes(drawn, barrier_names)

    if cur_idx >= len(barrier_names):
        st.success(f"✅ Tutte le {len(barrier_names)} barriere sono state definite.")
        render_frame_overlay_editor(
            target,
            mode="line",
            current_shape=None,
            confirmed_shapes=confirmed_shapes,
            current_label="Barriere confermate",
            instruction="Barriere completate. Usa Ridisegna tutto per ripartire.",
            stroke_color=_PALETTE[0],
            fill_color="rgba(255, 210, 50, 0.12)",
            disabled=True,
            key=f"{state_key}_editor_{selector_sig}",
        )

        st.markdown("**Barriere confermate**")
        for name in barrier_names:
            if name not in drawn:
                continue
            (x1, y1), (x2, y2) = drawn[name]
            st.write(f"- **{name}**: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})")

        if st.button("↺ Ridisegna tutto", key=k_final_reset):
            st.session_state[k_data] = {}
            st.session_state[k_idx] = 0

        return drawn

    cur_name = barrier_names[cur_idx]
    color = _PALETTE[cur_idx % len(_PALETTE)]

    st.info(
        f"**Barriera {cur_idx + 1} / {len(barrier_names)}: `{cur_name}`** "
        "Trascina una linea sopra il frame e rilascia per confermare. "
        "Azzera annulla l'ultima barriera."
    )

    event = render_frame_overlay_editor(
        target,
        mode="line",
        current_shape=None,
        confirmed_shapes=confirmed_shapes,
        current_label=cur_name,
        instruction="Trascina una linea sopra il frame. Rilascia per confermare. Usa Azzera per annullare l'ultima barriera.",
        stroke_color=color,
        fill_color="rgba(255, 210, 50, 0.12)",
        key=f"{state_key}_editor_{selector_sig}",
    )

    if isinstance(event, dict) and _is_new_event(event, f"{state_key}_{selector_sig}"):
        action = event.get("action")
        if action == "draw":
            shape = event.get("shape")
            if isinstance(shape, dict):
                drawn[cur_name] = _clamp_barrier(_to_barrier(shape), width, height)
                st.session_state[k_data] = drawn
                st.session_state[k_idx] = cur_idx + 1
        elif action == "clear":
            if cur_idx > 0:
                last_name = barrier_names[cur_idx - 1]
                drawn.pop(last_name, None)
                st.session_state[k_data] = drawn
                st.session_state[k_idx] = cur_idx - 1
            else:
                if drawn:
                    drawn.clear()
                    st.session_state[k_data] = drawn

    return drawn
