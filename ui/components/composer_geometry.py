"""Presentation component for video source, ROI, and barrier selection."""

from __future__ import annotations

import streamlit as st

from ui.components.barrier_selector import render_barrier_selector
from ui.components.roi_selector import render_roi_selector
from ui.components.video_selector import render_video_selector
from ui.services.pipeline_builder_service import builder_state, selected_resize, selected_signal_extractor
from ui.state import session


def render_video_and_geometry() -> None:
    """Render video selection and geometry controls with explicit state sync."""
    st.markdown("### Sorgente e geometria")
    video_path, first_frame, metadata = render_video_selector()

    if video_path:
        previous_video = session.get(session.VIDEO_PATH)
        if previous_video != video_path:
            _clear_geometry_outputs(clear_video=False)
        session.put(session.VIDEO_PATH, video_path)
        session.put(session.FIRST_FRAME, first_frame)
        session.put(session.FRAME_META, metadata)
    else:
        _clear_geometry_outputs(clear_video=True)
        st.info("Seleziona un video per abilitare ROI, barriere ed esecuzione.")
        return

    resize = selected_resize()
    _sync_resize_state(resize)
    barrier_names = builder_state().barrier_names
    _sync_barrier_names_state(barrier_names)

    frame = session.get(session.FIRST_FRAME)
    st.markdown("#### ROI di tracking")
    previous_roi = session.get(session.ROI_BOX)
    roi = render_roi_selector(frame, resize=resize, key="studio_roi")
    if roi is None:
        session.clear(session.ROI_BOX)
    else:
        session.put(session.ROI_BOX, roi)
        st.success(f"ROI selezionata: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    if previous_roi != roi:
        _clear_run_outputs()

    if selected_signal_extractor() == "opencv_multi_tracker":
        st.markdown("#### Barriere")
        previous_barriers = session.get(session.BARRIERS, {})
        barrier_state = render_barrier_selector(
            frame,
            barrier_names=list(barrier_names),
            resize=resize,
            state_key="studio_barriers",
        )
        session.put(session.BARRIER_SELECTION_STATE, barrier_state)
        if barrier_state.confirmed:
            session.put(session.BARRIERS, barrier_state.as_dict())
        else:
            session.clear(session.BARRIERS)
        if previous_barriers != barrier_state.as_dict():
            _clear_run_outputs()
        return

    had_barriers = bool(session.get(session.BARRIERS)) or session.get(session.BARRIER_SELECTION_STATE) is not None
    session.clear(session.BARRIER_SELECTION_STATE)
    session.clear(session.BARRIERS)
    if had_barriers:
        _clear_run_outputs()


def _sync_resize_state(resize: tuple[int, int] | None) -> None:
    last_resize_key = "studio_last_resize"
    previous_resize = st.session_state.get(last_resize_key, resize)
    if previous_resize != resize:
        _clear_geometry_outputs(clear_video=False)
    st.session_state[last_resize_key] = resize


def _sync_barrier_names_state(barrier_names: tuple[str, ...]) -> None:
    last_barrier_names_key = "studio_last_barrier_names"
    previous_barrier_names = st.session_state.get(last_barrier_names_key, barrier_names)
    if previous_barrier_names != barrier_names:
        session.clear(session.BARRIERS)
        session.clear(session.BARRIER_SELECTION_STATE)
        session.clear(session.PIPELINE_OUTPUTS)
        session.clear(session.TRACKING_VIDEO_CACHE)
    st.session_state[last_barrier_names_key] = barrier_names


def _clear_geometry_outputs(*, clear_video: bool) -> None:
    for key in (
        session.ROI_BOX,
        session.BARRIERS,
        session.BARRIER_SELECTION_STATE,
    ):
        session.clear(key)
    _clear_run_outputs()

    if clear_video:
        for key in (
            session.VIDEO_PATH,
            session.FIRST_FRAME,
            session.FRAME_META,
        ):
            session.clear(key)


def _clear_run_outputs() -> None:
    for key in (
        session.PIPELINE_OUTPUTS,
        session.TRACKING_VIDEO_CACHE,
    ):
        session.clear(key)
