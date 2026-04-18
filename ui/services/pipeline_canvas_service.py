"""
View-model builder for the visual pipeline canvas.

This module translates the current pipeline configuration and observable core
runtime data into a presentation-only graph structure. It keeps business logic
out of the Streamlit page and gives the canvas renderer a stable contract.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from library.core.events.Event import Event
from library.core.events.PipelineEvent import PipelineEvent
from library.core.pipeline.PipelineRunSnapshot import PipelineRunSnapshot, PipelineRunState
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry
from ui.components.pipeline_canvas_models import (
    CanvasEdge,
    CanvasNode,
    CanvasPort,
    EdgeKind,
    NodeCategory,
    NodeDetails,
    NodeState,
    PipelineCanvasModel,
    PortDataType,
    PortDirection,
)
from ui.state.canvas import layout as canvas_layout
from ui.state.canvas import viewport as canvas_viewport

_MAIN_STAGE_KEYS = (
    "frame_extractor",
    "frame_cleaners",
    "signal_extractor",
    "signal_cleaners",
    "analyzers",
    "visualizers",
)


def build_pipeline_canvas_model(
    config: dict[str, Any],
    registry: PluginRegistry,
    selected_stage: str,
    runtime_issues: list[str],
    run_snapshots: list[PipelineRunSnapshot],
    recent_events: list[Event],
) -> PipelineCanvasModel:
    """
    Build the graph consumed by the interactive pipeline designer canvas.

    The returned model is strictly presentation data:
    - nodes expose visual stage metadata
    - ports expose contract-level types
    - edges expose the allowed flow kinds and labels
    """
    pipeline = dict(config.get("pipeline", {}))
    issue_map = _group_runtime_issues(runtime_issues)
    stage_events = _collect_stage_events(pipeline, registry, recent_events)
    stage_positions = canvas_layout()
    viewport = canvas_viewport()

    latest_snapshot = _latest_snapshot(run_snapshots, predicate=lambda _: True)

    nodes = [
        _frame_extractor_node(
            pipeline,
            selected_stage,
            stage_positions,
            issue_map,
            latest_snapshot,
            stage_events.get("frame_extractor", ()),
        ),
        _frame_cleaners_node(
            pipeline,
            selected_stage,
            stage_positions,
            issue_map,
            latest_snapshot,
            stage_events.get("frame_cleaners", ()),
        ),
        _signal_extractor_node(
            pipeline,
            selected_stage,
            stage_positions,
            issue_map,
            latest_snapshot,
            stage_events.get("signal_extractor", ()),
        ),
        _signal_cleaners_node(
            pipeline,
            selected_stage,
            stage_positions,
            issue_map,
            latest_snapshot,
            stage_events.get("signal_cleaners", ()),
        ),
        _analyzers_node(
            pipeline,
            selected_stage,
            stage_positions,
            issue_map,
            latest_snapshot,
            stage_events.get("analyzers", ()),
        ),
        _visualizers_node(
            pipeline,
            selected_stage,
            stage_positions,
            issue_map,
            latest_snapshot,
            stage_events.get("visualizers", ()),
        ),
    ]

    edges = list(_main_pipeline_edges())
    branch_nodes, branch_edges = _branch_nodes_and_edges(
        selected_stage=selected_stage,
        issue_map=issue_map,
        run_snapshots=run_snapshots,
        recent_events=recent_events,
        stage_events=stage_events,
    )
    nodes.extend(branch_nodes)
    edges.extend(branch_edges)

    return PipelineCanvasModel(
        nodes=tuple(nodes),
        edges=tuple(edges),
        initial_pan_x=viewport.pan_x,
        initial_pan_y=viewport.pan_y,
        initial_zoom=viewport.zoom,
    )


def _frame_extractor_node(
    pipeline: dict[str, Any],
    selected_stage: str,
    positions: dict[str, dict[str, int]],
    issue_map: dict[str, list[str]],
    snapshot: PipelineRunSnapshot | None,
    emitted_events: tuple[str, ...],
) -> CanvasNode:
    extractor = dict(pipeline.get("frame_extractor", {}))
    params = dict(extractor.get("params", {}))
    config = dict(params.get("config", {}))
    path = str(params.get("path", ""))
    preview = Path(path).name if path else "no video selected"
    ports = (
        _port("frame_extractor", "video_in", "Video", PortDirection.INPUT, PortDataType.VIDEO),
        _port("frame_extractor", "frames_out", "FrameBuffer", PortDirection.OUTPUT, PortDataType.FRAME),
    )
    return CanvasNode(
        node_id="frame_extractor",
        stage_key="frame_extractor",
        stage_type="FrameExtractor",
        title="Frame extractor",
        category=NodeCategory.SOURCE,
        state=_node_state(True, snapshot),
        components=(str(extractor.get("name", "unconfigured")),),
        expected_output="FrameBuffer",
        details=NodeDetails(
            input_types=("VideoSource",),
            output_types=("FrameBuffer",),
            emitted_events=emitted_events,
            configuration={
                "path": preview,
                "resize": config.get("resize"),
                "stride": config.get("stride"),
                "max_frames": config.get("max_frames"),
            },
        ),
        ports=ports,
        preview=f"source: {preview}",
        position=_position_for("frame_extractor", positions),
        warnings=tuple(issue_map["frame_extractor"]),
        selected=selected_stage == "frame_extractor",
    )


def _frame_cleaners_node(
    pipeline: dict[str, Any],
    selected_stage: str,
    positions: dict[str, dict[str, int]],
    issue_map: dict[str, list[str]],
    snapshot: PipelineRunSnapshot | None,
    emitted_events: tuple[str, ...],
) -> CanvasNode:
    cleaners = [dict(item) for item in pipeline.get("frame_cleaners", [])]
    component_names = tuple(item.get("name", "unnamed") for item in cleaners) or ("none",)
    active = bool(cleaners)
    return CanvasNode(
        node_id="frame_cleaners",
        stage_key="frame_cleaners",
        stage_type="FrameCleanerStage",
        title="Frame cleaners",
        category=NodeCategory.TRANSFORM,
        state=_node_state(active, snapshot),
        components=component_names,
        expected_output="FrameBuffer",
        details=NodeDetails(
            input_types=("FrameBuffer",),
            output_types=("FrameBuffer",),
            emitted_events=emitted_events,
            configuration=_named_component_configuration(cleaners),
        ),
        ports=(
            _port("frame_cleaners", "frames_in", "FrameBuffer", PortDirection.INPUT, PortDataType.FRAME),
            _port("frame_cleaners", "frames_out", "CleanFrameBuffer", PortDirection.OUTPUT, PortDataType.FRAME),
        ),
        preview=f"{len(cleaners)} cleaner active" if cleaners else "pass-through stage",
        position=_position_for("frame_cleaners", positions),
        warnings=tuple(issue_map["frame_cleaners"]),
        selected=selected_stage == "frame_cleaners",
    )


def _signal_extractor_node(
    pipeline: dict[str, Any],
    selected_stage: str,
    positions: dict[str, dict[str, int]],
    issue_map: dict[str, list[str]],
    snapshot: PipelineRunSnapshot | None,
    emitted_events: tuple[str, ...],
) -> CanvasNode:
    extractor = dict(pipeline.get("signal_extractor", {}))
    params = dict(extractor.get("params", {}))
    component_name = str(extractor.get("name", "unconfigured"))
    ports = [
        _port("signal_extractor", "frames_in", "FrameBuffer", PortDirection.INPUT, PortDataType.FRAME),
        _port("signal_extractor", "signal_out", "Signal", PortDirection.OUTPUT, PortDataType.SIGNAL),
    ]
    if emitted_events:
        ports.append(
            _port("signal_extractor", "events_out", "Event", PortDirection.OUTPUT, PortDataType.EVENT, required=False)
        )
    return CanvasNode(
        node_id="signal_extractor",
        stage_key="signal_extractor",
        stage_type="SignalExtractor",
        title="Signal extractor",
        category=NodeCategory.SIGNAL,
        state=_node_state(True, snapshot),
        components=(component_name,),
        expected_output=_signal_expected_output(component_name),
        details=NodeDetails(
            input_types=("FrameBuffer",),
            output_types=(_signal_expected_output(component_name),),
            emitted_events=emitted_events,
            configuration=params,
        ),
        ports=tuple(ports),
        preview=_signal_preview(component_name),
        position=_position_for("signal_extractor", positions),
        warnings=tuple(issue_map["signal_extractor"]),
        selected=selected_stage == "signal_extractor",
    )


def _signal_cleaners_node(
    pipeline: dict[str, Any],
    selected_stage: str,
    positions: dict[str, dict[str, int]],
    issue_map: dict[str, list[str]],
    snapshot: PipelineRunSnapshot | None,
    emitted_events: tuple[str, ...],
) -> CanvasNode:
    cleaners = [dict(item) for item in pipeline.get("signal_cleaners", [])]
    component_names = tuple(item.get("name", "unnamed") for item in cleaners) or ("none",)
    active = bool(cleaners)
    return CanvasNode(
        node_id="signal_cleaners",
        stage_key="signal_cleaners",
        stage_type="SignalCleanerStage",
        title="Signal cleaners",
        category=NodeCategory.TRANSFORM,
        state=_node_state(active, snapshot),
        components=component_names,
        expected_output="Signal",
        details=NodeDetails(
            input_types=("Signal",),
            output_types=("Signal",),
            emitted_events=emitted_events,
            configuration=_named_component_configuration(cleaners),
        ),
        ports=(
            _port("signal_cleaners", "signal_in", "Signal", PortDirection.INPUT, PortDataType.SIGNAL),
            _port("signal_cleaners", "signal_out", "CleanSignal", PortDirection.OUTPUT, PortDataType.SIGNAL),
        ),
        preview=f"{len(cleaners)} transform active" if cleaners else "pass-through stage",
        position=_position_for("signal_cleaners", positions),
        warnings=tuple(issue_map["signal_cleaners"]),
        selected=selected_stage == "signal_cleaners",
    )


def _analyzers_node(
    pipeline: dict[str, Any],
    selected_stage: str,
    positions: dict[str, dict[str, int]],
    issue_map: dict[str, list[str]],
    snapshot: PipelineRunSnapshot | None,
    emitted_events: tuple[str, ...],
) -> CanvasNode:
    analyzers = [dict(item) for item in pipeline.get("analyzers", [])]
    component_names = tuple(item.get("name", "unnamed") for item in analyzers) or ("none",)
    active = bool(analyzers)
    return CanvasNode(
        node_id="analyzers",
        stage_key="analyzers",
        stage_type="AnalyzerStage",
        title="Analyzers",
        category=NodeCategory.ANALYTICS,
        state=_node_state(active, snapshot),
        components=component_names,
        expected_output="AnalysisResult[]",
        details=NodeDetails(
            input_types=("Signal",),
            output_types=("AnalysisResult[]",),
            emitted_events=emitted_events,
            configuration=_named_component_configuration(analyzers),
        ),
        ports=(
            _port("analyzers", "signal_in", "Signal", PortDirection.INPUT, PortDataType.SIGNAL),
            _port("analyzers", "analysis_out", "Analysis", PortDirection.OUTPUT, PortDataType.ANALYSIS),
        ),
        preview=f"{len(analyzers)} output planned" if analyzers else "no analytics configured",
        position=_position_for("analyzers", positions),
        warnings=tuple(issue_map["analyzers"]),
        selected=selected_stage == "analyzers",
    )


def _visualizers_node(
    pipeline: dict[str, Any],
    selected_stage: str,
    positions: dict[str, dict[str, int]],
    issue_map: dict[str, list[str]],
    snapshot: PipelineRunSnapshot | None,
    emitted_events: tuple[str, ...],
) -> CanvasNode:
    visualizers = [dict(item) for item in pipeline.get("visualizers", [])]
    component_names = tuple(item.get("name", "unnamed") for item in visualizers) or ("ui-results",)
    active = bool(visualizers)
    target_preview = _visualizer_targets_preview(visualizers)
    return CanvasNode(
        node_id="visualizers",
        stage_key="visualizers",
        stage_type="VisualizerStage",
        title="Visualizers",
        category=NodeCategory.PRESENTATION,
        state=_node_state(True, snapshot if active else None),
        components=component_names,
        expected_output="RenderedView",
        details=NodeDetails(
            input_types=("AnalysisResult[]",),
            output_types=("RenderedView",),
            emitted_events=emitted_events,
            configuration=_named_component_configuration(visualizers),
        ),
        ports=(
            _port("visualizers", "analysis_in", "Analysis", PortDirection.INPUT, PortDataType.ANALYSIS),
            _port("visualizers", "view_out", "View", PortDirection.OUTPUT, PortDataType.VIEW, required=False),
        ),
        preview=target_preview,
        position=_position_for("visualizers", positions),
        warnings=tuple(issue_map["visualizers"]),
        selected=selected_stage == "visualizers",
    )


def _branch_nodes_and_edges(
    *,
    selected_stage: str,
    issue_map: dict[str, list[str]],
    run_snapshots: list[PipelineRunSnapshot],
    recent_events: list[Event],
    stage_events: dict[str, tuple[str, ...]],
) -> tuple[list[CanvasNode], list[CanvasEdge]]:
    trigger_events = [event for event in recent_events if event.event_type == PipelineEvent.event_type]
    if not trigger_events and not any(stage_events.values()):
        return [], []

    nodes: list[CanvasNode] = []
    edges: list[CanvasEdge] = []

    trigger_ids = tuple(
        str(event.payload.get("pipeline_id", "secondary"))
        for event in trigger_events[-3:]
    ) or ("pending trigger",)
    branch_snapshot = _latest_snapshot(
        run_snapshots,
        predicate=lambda snap: snap.pipeline_id.startswith("secondary-"),
    )

    gateway_node = CanvasNode(
        node_id="branch_trigger",
        stage_key="branch_trigger",
        stage_type="PipelineEvent",
        title="Event trigger bus",
        category=NodeCategory.EVENT,
        state=_node_state(True, branch_snapshot),
        components=trigger_ids,
        expected_output="PipelineTrigger",
        details=NodeDetails(
            input_types=("Event",),
            output_types=("PipelineTrigger",),
            emitted_events=tuple(sorted({event.event_type for event in trigger_events})),
            configuration={"recent_triggers": list(trigger_ids)},
        ),
        ports=(
            _port("branch_trigger", "event_in", "Event", PortDirection.INPUT, PortDataType.EVENT),
            _port("branch_trigger", "trigger_out", "Trigger", PortDirection.OUTPUT, PortDataType.EVENT),
        ),
        preview="branch orchestration",
        position=(1100, 520),
        warnings=tuple(issue_map.get("branch_trigger", [])),
        selected=selected_stage == "branch_trigger",
    )
    nodes.append(gateway_node)

    secondary_node = CanvasNode(
        node_id="secondary_pipeline",
        stage_key="secondary_pipeline",
        stage_type="SecondaryPipeline",
        title="Secondary pipeline",
        category=NodeCategory.EVENT,
        state=_node_state(True, branch_snapshot),
        components=trigger_ids,
        expected_output="Async pipeline branch",
        details=NodeDetails(
            input_types=("PipelineTrigger",),
            output_types=("AnalysisResult[]",),
            emitted_events=(),
            configuration={"pipeline_ids": list(trigger_ids)},
        ),
        ports=(
            _port("secondary_pipeline", "trigger_in", "Trigger", PortDirection.INPUT, PortDataType.EVENT),
            _port("secondary_pipeline", "analysis_out", "Analysis", PortDirection.OUTPUT, PortDataType.ANALYSIS, required=False),
        ),
        preview="event-driven branch",
        position=(1450, 540),
        warnings=(),
        selected=selected_stage == "secondary_pipeline",
    )
    nodes.append(secondary_node)

    for stage_key, events in stage_events.items():
        if not events:
            continue
        node_id = stage_key
        source_port_id = f"{stage_key}:events_out"
        edges.append(
            CanvasEdge(
                edge_id=f"{node_id}->branch_trigger:event",
                source_node_id=node_id,
                source_port_id=source_port_id,
                target_node_id="branch_trigger",
                target_port_id="branch_trigger:event_in",
                label=events[0],
                kind=EdgeKind.EVENT,
            )
        )

    edges.append(
        CanvasEdge(
            edge_id="branch_trigger->secondary_pipeline",
            source_node_id="branch_trigger",
            source_port_id="branch_trigger:trigger_out",
            target_node_id="secondary_pipeline",
            target_port_id="secondary_pipeline:trigger_in",
            label="PipelineTrigger",
            kind=EdgeKind.SECONDARY,
        )
    )
    return nodes, edges


def _main_pipeline_edges() -> Iterable[CanvasEdge]:
    yield CanvasEdge(
        edge_id="frame_extractor->frame_cleaners",
        source_node_id="frame_extractor",
        source_port_id="frame_extractor:frames_out",
        target_node_id="frame_cleaners",
        target_port_id="frame_cleaners:frames_in",
        label="FrameBuffer",
        kind=EdgeKind.MAIN,
    )
    yield CanvasEdge(
        edge_id="frame_cleaners->signal_extractor",
        source_node_id="frame_cleaners",
        source_port_id="frame_cleaners:frames_out",
        target_node_id="signal_extractor",
        target_port_id="signal_extractor:frames_in",
        label="CleanFrameBuffer",
        kind=EdgeKind.MAIN,
    )
    yield CanvasEdge(
        edge_id="signal_extractor->signal_cleaners",
        source_node_id="signal_extractor",
        source_port_id="signal_extractor:signal_out",
        target_node_id="signal_cleaners",
        target_port_id="signal_cleaners:signal_in",
        label="Signal",
        kind=EdgeKind.MAIN,
    )
    yield CanvasEdge(
        edge_id="signal_cleaners->analyzers",
        source_node_id="signal_cleaners",
        source_port_id="signal_cleaners:signal_out",
        target_node_id="analyzers",
        target_port_id="analyzers:signal_in",
        label="CleanSignal",
        kind=EdgeKind.MAIN,
    )
    yield CanvasEdge(
        edge_id="analyzers->visualizers",
        source_node_id="analyzers",
        source_port_id="analyzers:analysis_out",
        target_node_id="visualizers",
        target_port_id="visualizers:analysis_in",
        label="Analysis",
        kind=EdgeKind.MAIN,
    )


def _collect_stage_events(
    pipeline: dict[str, Any],
    registry: PluginRegistry,
    recent_events: list[Event],
) -> dict[str, tuple[str, ...]]:
    component_sources: dict[str, set[str]] = defaultdict(set)
    stage_plugins = {
        "frame_extractor": [(PluginCategory.FRAME_EXTRACTOR, pipeline.get("frame_extractor", {}).get("name"))],
        "frame_cleaners": [(PluginCategory.FRAME_CLEANER, item.get("name")) for item in pipeline.get("frame_cleaners", [])],
        "signal_extractor": [(PluginCategory.SIGNAL_EXTRACTOR, pipeline.get("signal_extractor", {}).get("name"))],
        "signal_cleaners": [(PluginCategory.SIGNAL_CLEANER, item.get("name")) for item in pipeline.get("signal_cleaners", [])],
        "analyzers": [(PluginCategory.ANALYZER, item.get("name")) for item in pipeline.get("analyzers", [])],
        "visualizers": [(PluginCategory.VISUALIZER, item.get("name")) for item in pipeline.get("visualizers", [])],
    }

    for stage_key, entries in stage_plugins.items():
        for category, plugin_name in entries:
            if not plugin_name:
                continue
            try:
                component_sources[stage_key].add(registry.get(category, str(plugin_name)).factory.__name__)
            except Exception:
                continue

    stage_events: dict[str, set[str]] = defaultdict(set)
    for event in recent_events:
        if event.event_type.startswith("pipeline.") or event.event_type == PipelineEvent.event_type:
            continue
        for stage_key, source_names in component_sources.items():
            if event.source in source_names:
                stage_events[stage_key].add(event.event_type)
    return {stage_key: tuple(sorted(values)) for stage_key, values in stage_events.items()}


def _group_runtime_issues(issues: list[str]) -> dict[str, list[str]]:
    issue_map = {stage: [] for stage in (*_MAIN_STAGE_KEYS, "branch_trigger", "secondary_pipeline")}
    for issue in issues:
        lowered = issue.lower()
        if "video" in lowered:
            issue_map["frame_extractor"].append(issue)
        elif "roi" in lowered:
            issue_map["signal_extractor"].append(issue)
        elif "barriera" in lowered:
            issue_map["signal_extractor"].append(issue)
            issue_map["analyzers"].append(issue)
        elif "analyzer" in lowered:
            issue_map["analyzers"].append(issue)
        elif "opencv_gray" in lowered:
            issue_map["frame_cleaners"].append(issue)
    return issue_map


def _latest_snapshot(
    snapshots: list[PipelineRunSnapshot],
    predicate,
) -> PipelineRunSnapshot | None:
    matches = [snapshot for snapshot in snapshots if predicate(snapshot)]
    if not matches:
        return None
    return max(
        matches,
        key=lambda snap: (
            snap.completed_at or 0.0,
            snap.started_at or 0.0,
            snap.submitted_at or 0.0,
        ),
    )


def _node_state(configured: bool, snapshot: PipelineRunSnapshot | None) -> NodeState:
    if not configured:
        return NodeState.IDLE
    if snapshot is None:
        return NodeState.CONFIGURED
    if snapshot.state == PipelineRunState.RUNNING:
        return NodeState.RUNNING
    if snapshot.state == PipelineRunState.SUCCEEDED:
        return NodeState.COMPLETED
    if snapshot.state == PipelineRunState.FAILED:
        return NodeState.ERROR
    return NodeState.CONFIGURED


def _position_for(stage_key: str, positions: dict[str, dict[str, int]]) -> tuple[int, int]:
    raw = positions.get(stage_key, {"x": 0, "y": 0})
    return int(raw.get("x", 0)), int(raw.get("y", 0))


def _signal_expected_output(component_name: str) -> str:
    if component_name == "dense_optical_flow":
        return "DenseMotionSignal"
    if component_name == "opencv_multi_tracker":
        return "MultiObjectSignal"
    return "Signal"


def _signal_preview(component_name: str) -> str:
    if component_name == "dense_optical_flow":
        return "dense field + vector cells"
    if component_name == "opencv_multi_tracker":
        return "multi-object tracking stream"
    return "single-object tracking stream"


def _visualizer_targets_preview(visualizers: list[dict[str, Any]]) -> str:
    if not visualizers:
        return "results rendered by UI only"
    targeted = [
        item.get("result_indices")
        for item in visualizers
        if item.get("result_indices") is not None
    ]
    if not targeted:
        return f"{len(visualizers)} renderer(s) on all results"
    return f"{len(visualizers)} renderer(s) with targeted outputs"


def _named_component_configuration(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve the full list configuration instead of collapsing it to params only."""
    return {
        "items": [
            {
                "name": item.get("name", "unnamed"),
                "params": item.get("params", {}),
                **({"result_indices": item.get("result_indices")} if item.get("result_indices") is not None else {}),
            }
            for item in items
        ]
    }


def _port(
    node_id: str,
    suffix: str,
    label: str,
    direction: PortDirection,
    data_type: PortDataType,
    *,
    required: bool = True,
) -> CanvasPort:
    return CanvasPort(
        port_id=f"{node_id}:{suffix}",
        node_id=node_id,
        label=label,
        direction=direction,
        data_type=data_type,
        required=required,
    )
