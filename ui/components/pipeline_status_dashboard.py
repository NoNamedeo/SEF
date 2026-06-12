"""Pipeline status dashboard for async runs and branching output.

The dashboard summarizes snapshot state, recent lifecycle/domain events and
secondary pipeline activity so the UI shows what is happening without
requiring users to inspect raw logs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import streamlit as st

from sef.core.events.Event import Event
from sef.core.pipeline.PipelineRunSnapshot import PipelineRunSnapshot, PipelineRunState

_STATE_LABELS = {
    PipelineRunState.QUEUED: "Queued",
    PipelineRunState.RUNNING: "Running",
    PipelineRunState.SUCCEEDED: "Succeeded",
    PipelineRunState.FAILED: "Failed",
    PipelineRunState.CANCELLED: "Cancelled",
}

_STATE_EMOJI = {
    PipelineRunState.QUEUED: "🟡",
    PipelineRunState.RUNNING: "🔵",
    PipelineRunState.SUCCEEDED: "🟢",
    PipelineRunState.FAILED: "🔴",
    PipelineRunState.CANCELLED: "⚪",
}


def render_pipeline_status_dashboard(
    snapshots: list[PipelineRunSnapshot],
    events: list[Event],
    *,
    title: str = "Pipeline status",
    empty_message: str = "Nessuna pipeline tracciata in questa sessione.",
) -> None:
    """Render a concise overview of pipeline state and recent events."""
    st.markdown(f"### {title}")

    if not snapshots:
        st.info(empty_message)
        return

    event_groups = _group_events_by_pipeline(events)
    lifecycle_events = [event for event in events if str(event.event_type).startswith("pipeline.")]
    domain_events = [event for event in events if not str(event.event_type).startswith("pipeline.")]
    counts = Counter(snapshot.state for snapshot in snapshots)
    secondary_count = sum(1 for snapshot in snapshots if snapshot.pipeline_id.startswith("secondary-"))
    domain_count = len(domain_events)
    lifecycle_count = len(lifecycle_events)

    cols = st.columns(7)
    cols[0].metric("Total", len(snapshots))
    cols[1].metric("Queued", counts.get(PipelineRunState.QUEUED, 0))
    cols[2].metric("Running", counts.get(PipelineRunState.RUNNING, 0))
    cols[3].metric("Succeeded", counts.get(PipelineRunState.SUCCEEDED, 0))
    cols[4].metric("Failed", counts.get(PipelineRunState.FAILED, 0))
    cols[5].metric("Cancelled", counts.get(PipelineRunState.CANCELLED, 0))
    cols[6].metric("Secondary", secondary_count)

    c1, c2 = st.columns(2)
    c1.metric("Lifecycle events", lifecycle_count)
    c2.metric("Domain events", domain_count)

    ordered = sorted(
        snapshots,
        key=lambda snap: (
            _state_rank(snap.state),
            snap.completed_at or snap.started_at or snap.submitted_at or 0.0,
            snap.pipeline_id,
        ),
    )

    for snapshot in ordered:
        _render_snapshot_card(snapshot, event_groups.get(snapshot.pipeline_id, []))

    with st.expander("Domain events", expanded=bool(domain_events)):
        if not domain_events:
            st.info("Nessun domain event registrato.")
        else:
            _render_event_list(domain_events, limit=None)


def render_event_timeline(events: list[Event], *, limit: int = 20) -> None:
    """Render a compact recent-event timeline with pipeline grouping."""
    st.markdown("### Event timeline")
    if not events:
        st.info("Nessun evento registrato.")
        return

    pipeline_ids = sorted({event.payload.get("pipeline_id", "-") for event in events})
    selected = st.selectbox("Pipeline filter", ["All", *pipeline_ids], index=0, key="pipeline_event_filter")

    filtered = events
    if selected != "All":
        filtered = [event for event in events if event.payload.get("pipeline_id", "-") == selected]

    recent = list(reversed(filtered[-limit:]))
    if not recent:
        st.info("Nessun evento per il filtro selezionato.")
        return

    for event in recent:
        pipeline_id = event.payload.get("pipeline_id", "-")
        with st.container(border=True):
            st.write(f"**{event.event_type}**")
            st.caption(f"from {event.source} · pipeline={pipeline_id} · {_fmt_time(event.timestamp)}")


def _render_event_list(events: list[Event], limit: int | None = 20) -> None:
    subset = events if limit is None else events[-limit:]
    recent = list(reversed(subset))
    if not recent:
        st.info("Nessun evento da mostrare.")
        return

    for event in recent:
        pipeline_id = event.payload.get("pipeline_id", "-")
        with st.container(border=True):
            st.write(f"**{event.event_type}**")
            st.caption(f"from {event.source} · pipeline={pipeline_id} · {_fmt_time(event.timestamp)}")


def _render_snapshot_card(snapshot: PipelineRunSnapshot, events: list[Event]) -> None:
    label = _STATE_LABELS.get(snapshot.state, snapshot.state.value.title())
    emoji = _STATE_EMOJI.get(snapshot.state, "•")
    pipeline_kind = "secondary" if snapshot.pipeline_id.startswith("secondary-") else "primary"
    subtitle = f"{pipeline_kind} · attempt {snapshot.attempt}"

    with st.container(border=True):
        st.markdown(f"**{emoji} `{snapshot.pipeline_id}`**")
        st.caption(f"{label} · {subtitle}")

        cols = st.columns(4)
        cols[0].metric("Submitted", _fmt_time(snapshot.submitted_at))
        cols[1].metric("Started", _fmt_time(snapshot.started_at))
        cols[2].metric("Completed", _fmt_time(snapshot.completed_at))
        cols[3].metric("Events", len(events))

        if snapshot.error:
            st.error(snapshot.error)

        if events:
            with st.expander("Recent events", expanded=False):
                for event in events[-5:]:
                    st.write(f"- `{event.event_type}` · {event.source} · {_fmt_time(event.timestamp)}")


def _group_events_by_pipeline(events: Iterable[Event]) -> dict[str, list[Event]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        pipeline_id = str(event.payload.get("pipeline_id", "-"))
        grouped[pipeline_id].append(event)
    return grouped


def _state_rank(state: PipelineRunState) -> int:
    order = {
        PipelineRunState.RUNNING: 0,
        PipelineRunState.QUEUED: 1,
        PipelineRunState.FAILED: 2,
        PipelineRunState.CANCELLED: 3,
        PipelineRunState.SUCCEEDED: 4,
    }
    return order.get(state, 99)


def _fmt_time(value: float | None) -> str:
    if value is None:
        return "-"
    import time

    return time.strftime("%H:%M:%S", time.localtime(value))
