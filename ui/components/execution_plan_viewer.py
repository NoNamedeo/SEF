"""Streamlit components for readable execution-plan diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import streamlit as st

from ui.services.execution_plan_service import (
    format_execution_plan_text,
    summarize_execution_plan,
)


def render_execution_plan(
    execution_plan: dict[str, Any],
    *,
    title: str | None = "Execution plan",
) -> None:
    """Render stream/batch decisions, materialization boundaries and runtime policy."""
    if title:
        st.markdown(f"### {title}")
    if not execution_plan:
        st.info("Execution plan non disponibile.")
        return

    summary = summarize_execution_plan(execution_plan)
    cols = st.columns(6)
    cols[0].metric("Stages", summary["stage_count"])
    cols[1].metric("Streaming", summary["streaming_count"])
    cols[2].metric("Batch", summary["batch_count"])
    cols[3].metric("Materializzazioni", summary["materialization_count"])
    cols[4].metric("Frame parallel", summary["parallel_count"])
    cols[5].metric("Latency", summary["latency_policy"])

    streamable = "si" if summary["streamable_end_to_end"] else "no"
    st.caption(f"Stream end-to-end: {streamable}")

    _render_runtime_section(execution_plan)
    _render_materialization_boundaries(execution_plan)
    _render_stage_groups(execution_plan)

    with st.expander("Formato leggibile", expanded=True):
        st.code(format_execution_plan_text(execution_plan), language="text")


def _render_runtime_section(execution_plan: dict[str, Any]) -> None:
    runtime = dict(execution_plan.get("runtime", {}) or {})
    if not runtime:
        return

    latency_policy = dict(runtime.get("latency_policy", {}) or {})
    params = dict(latency_policy.get("params", {}) or {})
    with st.expander("Runtime streaming", expanded=True):
        cols = st.columns(4)
        cols[0].metric("Frame buffer", runtime.get("frame_buffer_size", "-"))
        cols[1].metric("Signal buffer", runtime.get("signal_buffer_size", "-"))
        cols[2].metric("Data buffer", runtime.get("data_buffer_size", "-"))
        cols[3].metric("Policy", latency_policy.get("name", "unknown"))
        if params:
            st.dataframe(
                [{"param": key, "value": value} for key, value in params.items()],
                hide_index=True,
                width="stretch",
            )


def _render_materialization_boundaries(execution_plan: dict[str, Any]) -> None:
    boundaries = list(execution_plan.get("materialization_boundaries", []) or [])
    if not boundaries:
        st.success("Nessun boundary di materializzazione nel piano corrente.")
        return
    st.warning(f"Boundary di materializzazione presenti: {len(boundaries)}")
    st.dataframe(
        [
            {
                "stage": item.get("stage_id"),
                "component": item.get("component_name"),
                "estimated materialized bytes": item.get("estimated_materialized_bytes"),
            }
            for item in boundaries
        ],
        hide_index=True,
        width="stretch",
    )


def _render_stage_groups(execution_plan: dict[str, Any]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stage in execution_plan.get("stages", []) or []:
        grouped[str(stage.get("stage_group", "unknown"))].append(dict(stage))

    if not grouped:
        st.info("Nessuno stage pianificato.")
        return

    st.markdown("#### Decisioni per stage")
    for group_name, stages in grouped.items():
        with st.expander(group_name.replace("_", " ").title(), expanded=True):
            st.dataframe(
                [_stage_row(stage) for stage in stages],
                hide_index=True,
                width="stretch",
            )


def _stage_row(stage: dict[str, Any]) -> dict[str, Any]:
    capabilities = dict(stage.get("capabilities", {}) or {})
    return {
        "stage": stage.get("stage_id"),
        "component": stage.get("component_name"),
        "mode": stage.get("execution_mode"),
        "materializes": bool(stage.get("materializes_input")),
        "parallel": bool(capabilities.get("supports_frame_parallelism")),
        "realtime": bool(capabilities.get("realtime_safe")),
        "stateful": bool(capabilities.get("stateful")),
        "queue bytes": stage.get("estimated_queue_bytes"),
        "materialized bytes": stage.get("estimated_materialized_bytes"),
        "reason": stage.get("reason"),
    }
