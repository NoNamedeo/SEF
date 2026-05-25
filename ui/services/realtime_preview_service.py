"""Application service for browser-compatible realtime previews."""

from __future__ import annotations

import copy
import threading
from typing import Any

from library.core.realtime.LatestRealtimeFrameStore import LatestRealtimeFrameStore, RealtimeFrameSnapshot

STREAMLIT_COCO_POSE_REALTIME_VISUALIZER = "streamlit_coco_pose_realtime"
STREAMLIT_REALTIME_VISUALIZERS = frozenset({STREAMLIT_COCO_POSE_REALTIME_VISUALIZER})

_store_lock = threading.Lock()
_stores: dict[str, LatestRealtimeFrameStore] = {}


def sink_for_id(sink_id: str | None) -> LatestRealtimeFrameStore:
    """Return the realtime sink associated with a pipeline or preview id."""
    key = _normalise_sink_id(sink_id)
    with _store_lock:
        store = _stores.get(key)
        if store is None:
            store = LatestRealtimeFrameStore()
            _stores[key] = store
        return store


def reset_sink(sink_id: str | None) -> None:
    """Prepare a sink for a new run."""
    sink_for_id(sink_id).reset()


def snapshot_for_id(sink_id: str | None) -> RealtimeFrameSnapshot:
    """Return the latest preview frame for a pipeline or preview id."""
    return sink_for_id(sink_id).snapshot()


def preview_has_content(sink_id: str | None) -> bool:
    """Return True when a sink has either an active producer or a retained frame."""
    snapshot = snapshot_for_id(sink_id)
    return snapshot.active or snapshot.frame is not None


def config_has_streamlit_realtime_visualizer(config: dict[str, Any]) -> bool:
    """Return True when the config contains a Streamlit realtime visualizer."""
    return bool(_streamlit_realtime_visualizer_names(config))


def streamlit_realtime_visualizer_names(config: dict[str, Any]) -> tuple[str, ...]:
    """Return configured browser-compatible realtime visualizer names."""
    return _streamlit_realtime_visualizer_names(config)


def with_realtime_sink_ids(config: dict[str, Any], sink_id: str) -> dict[str, Any]:
    """
    Return an execution config with sink ids injected into realtime visualizers.

    The generated composer config remains serializable and UI-neutral; this
    execution adapter supplies the runtime sink only when a run starts.
    """
    patched = copy.deepcopy(config)
    pipeline = patched.get("pipeline", {})
    if not isinstance(pipeline, dict):
        return patched

    _ensure_realtime_frame_tap(pipeline, sink_id)

    for visualizer in pipeline.get("visualizers", []) or []:
        if not isinstance(visualizer, dict):
            continue
        if str(visualizer.get("name", "")) not in STREAMLIT_REALTIME_VISUALIZERS:
            continue
        params = dict(visualizer.get("params", {}))
        params["sink_id"] = sink_id
        visualizer["params"] = params
    return patched


def _ensure_realtime_frame_tap(pipeline: dict[str, Any], sink_id: str) -> None:
    frame_processors = pipeline.setdefault("frame_processors", [])
    if not isinstance(frame_processors, list):
        return
    for processor in frame_processors:
        if isinstance(processor, dict) and str(processor.get("name", "")) == "realtime_frame_tap":
            params = dict(processor.get("params", {}))
            params["sink_id"] = sink_id
            processor["params"] = params
            processor["processor_type"] = "frame_buffer"
            return
    frame_processors.insert(
        0,
        {
            "name": "realtime_frame_tap",
            "processor_type": "frame_buffer",
            "params": {
                "config": {"publish_every_n_frames": 1},
                "sink_id": sink_id,
            },
        },
    )


def _streamlit_realtime_visualizer_names(config: dict[str, Any]) -> tuple[str, ...]:
    pipeline = config.get("pipeline", {})
    if not isinstance(pipeline, dict):
        return ()
    names = []
    for visualizer in pipeline.get("visualizers", []) or []:
        if isinstance(visualizer, dict) and str(visualizer.get("name", "")) in STREAMLIT_REALTIME_VISUALIZERS:
            names.append(str(visualizer["name"]))
    return tuple(names)


def _normalise_sink_id(sink_id: str | None) -> str:
    if sink_id is None or not str(sink_id).strip():
        return "default"
    return str(sink_id).strip()
