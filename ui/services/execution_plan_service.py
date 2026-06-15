"""UI-facing execution-plan service.

The Streamlit layer needs to preview stream/batch decisions without leaking
planner orchestration into page components. This module is the presentation
adapter around the core planner and keeps a small process-local cache so heavy
component construction is not repeated on every Streamlit rerun.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sef.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from sef.core.pipeline.PipelineExecutionPlanner import PipelineExecutionPlanner
from sef.core.plugins.PluginRegistry import PluginRegistry

_MAX_PLAN_CACHE_ITEMS = 12
_cache_lock = threading.Lock()
_plan_cache: OrderedDict[str, "ExecutionPlanPreview"] = OrderedDict()


@dataclass(frozen=True, slots=True)
class ExecutionPlanPreview:
    """Serializable execution-plan preview for UI rendering."""

    plan: dict[str, Any] | None
    error: str | None = None

    @property
    def available(self) -> bool:
        """Return True when the planner produced a readable plan."""
        return self.plan is not None and self.error is None


def build_execution_plan_preview(
    config: dict[str, Any],
    registry: PluginRegistry,
) -> ExecutionPlanPreview:
    """
    Build and cache a planner preview for the current UI config.

    The cache key includes both the normalized config and the registry
    signature, so runtime plugin registrations naturally invalidate stale plans.
    """
    key = _cache_key(config, registry)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    preflight_error = _preflight_plan_side_effects(config)
    if preflight_error is not None:
        preview = ExecutionPlanPreview(plan=None, error=preflight_error)
        _put_cached(key, preview)
        return preview

    try:
        context = ConfigPipelineBuilder(registry).build_context(deepcopy(config))
        plan = PipelineExecutionPlanner().build(context).as_dict()
        preview = ExecutionPlanPreview(plan=plan)
    except Exception as exc:
        preview = ExecutionPlanPreview(plan=None, error=str(exc))

    _put_cached(key, preview)
    return preview


def summarize_execution_plan(plan: dict[str, Any]) -> dict[str, int | bool | str]:
    """Return compact counts used by plan dashboards and run controls."""
    stages = list(plan.get("stages", []) or [])
    streaming_count = sum(1 for stage in stages if stage.get("execution_mode") == "streaming")
    batch_count = sum(1 for stage in stages if stage.get("execution_mode") == "batch")
    materialization_count = sum(1 for stage in stages if stage.get("materializes_input"))
    parallel_count = sum(
        1
        for stage in stages
        if dict(stage.get("capabilities", {}) or {}).get("supports_frame_parallelism")
    )
    runtime = dict(plan.get("runtime", {}) or {})
    latency_policy = dict(runtime.get("latency_policy", {}) or {})
    return {
        "stage_count": len(stages),
        "streaming_count": streaming_count,
        "batch_count": batch_count,
        "materialization_count": materialization_count,
        "parallel_count": parallel_count,
        "streamable_end_to_end": bool(plan.get("streamable_end_to_end")),
        "latency_policy": str(latency_policy.get("name", "unknown")),
    }


def format_execution_plan_text(plan: dict[str, Any]) -> str:
    """Format the planner output as a stable, terminal-friendly summary."""
    summary = summarize_execution_plan(plan)
    lines = [
        "Pipeline execution plan",
        f"- streamable_end_to_end={summary['streamable_end_to_end']}",
        f"- latency_policy={summary['latency_policy']}",
        (
            "- stages="
            f"{summary['stage_count']} "
            f"(streaming={summary['streaming_count']}, batch={summary['batch_count']}, "
            f"materialization_boundaries={summary['materialization_count']})"
        ),
    ]

    for stage in plan.get("stages", []) or []:
        capabilities = dict(stage.get("capabilities", {}) or {})
        flags = []
        if stage.get("materializes_input"):
            flags.append("materializes_input")
        if capabilities.get("supports_frame_parallelism"):
            flags.append("frame_parallel")
        if capabilities.get("realtime_safe"):
            flags.append("realtime_safe")
        suffix = f" ({', '.join(flags)})" if flags else ""
        reason = stage.get("reason")
        lines.append(
            f"- {stage.get('stage_id')}: {stage.get('component_name')} "
            f"[{stage.get('execution_mode')}]{suffix}"
            + (f" - {reason}" if reason else "")
        )
    return "\n".join(lines)


def _get_cached(key: str) -> ExecutionPlanPreview | None:
    with _cache_lock:
        cached = _plan_cache.get(key)
        if cached is None:
            return None
        _plan_cache.move_to_end(key)
        return deepcopy(cached)


def _put_cached(key: str, value: ExecutionPlanPreview) -> None:
    with _cache_lock:
        _plan_cache[key] = deepcopy(value)
        _plan_cache.move_to_end(key)
        while len(_plan_cache) > _MAX_PLAN_CACHE_ITEMS:
            _plan_cache.popitem(last=False)


def _cache_key(config: dict[str, Any], registry: PluginRegistry) -> str:
    payload = {
        "config": config,
        "registry": _registry_signature(registry),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _preflight_plan_side_effects(config: dict[str, Any]) -> str | None:
    """
    Prevent plan previews from triggering model downloads.

    The runtime can still decide how to handle missing assets, but a UI preview
    must remain read-only and fast.
    """
    pipeline = dict(config.get("pipeline", {}) or {})
    signal_extractor = dict(pipeline.get("signal_extractor", {}) or {})
    if signal_extractor.get("name") != "yolo_coco_pose":
        return None
    params = dict(signal_extractor.get("params", {}) or {})
    model_name = str(params.get("model_name", "yolo11s-pose.pt"))
    model_path = Path(__file__).resolve().parents[2] / "sef" / "builtin" / "YOLOPoseModels" / model_name
    if model_path.exists():
        return None
    return (
        f"Modello YOLO non trovato per il preview plan: {model_path}. "
        "Aggiungi il file modello o scegli un modello gia presente."
    )


def _registry_signature(registry: PluginRegistry) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (
                str(plugin.category),
                plugin.name,
                plugin.factory_path,
            )
            for plugin in registry.list()
        )
    )
