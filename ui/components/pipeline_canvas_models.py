"""
Typed view-model for the visual pipeline canvas.

These dataclasses describe only presentation-oriented graph data. They do not
build or execute pipelines; they expose a stable structure that the UI renderer
can consume without prop drilling or ad-hoc dictionaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class NodeCategory(StrEnum):
    SOURCE = "source"
    TRANSFORM = "transform"
    SIGNAL = "signal"
    ANALYTICS = "analytics"
    PRESENTATION = "presentation"
    EVENT = "event"


class NodeState(StrEnum):
    IDLE = "idle"
    CONFIGURED = "configured"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class PortDirection(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class PortDataType(StrEnum):
    VIDEO = "video"
    FRAME = "frame"
    SIGNAL = "signal"
    EVENT = "event"
    ANALYSIS = "analysis"
    VIEW = "view"


class EdgeKind(StrEnum):
    MAIN = "main"
    SECONDARY = "secondary"
    EVENT = "event"


@dataclass(frozen=True, slots=True)
class NodeDetails:
    """Expanded node metadata shown inside the canvas details panel."""

    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    emitted_events: tuple[str, ...] = ()
    configuration: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CanvasPort:
    """Connection point rendered on a node boundary."""

    port_id: str
    node_id: str
    label: str
    direction: PortDirection
    data_type: PortDataType
    required: bool = True


@dataclass(frozen=True, slots=True)
class CanvasNode:
    """Visual stage node in the pipeline designer."""

    node_id: str
    stage_key: str
    stage_type: str
    title: str
    category: NodeCategory
    state: NodeState
    components: tuple[str, ...]
    expected_output: str
    details: NodeDetails
    ports: tuple[CanvasPort, ...]
    preview: str | None = None
    position: tuple[int, int] = (0, 0)
    warnings: tuple[str, ...] = ()
    selected: bool = False


@dataclass(frozen=True, slots=True)
class CanvasEdge:
    """Directional connection between two compatible ports."""

    edge_id: str
    source_node_id: str
    source_port_id: str
    target_node_id: str
    target_port_id: str
    label: str
    kind: EdgeKind


@dataclass(frozen=True, slots=True)
class PipelineCanvasModel:
    """Whole-canvas payload used by the HTML renderer."""

    nodes: tuple[CanvasNode, ...]
    edges: tuple[CanvasEdge, ...]
    surface_width: int = 3800
    surface_height: int = 1400
    initial_pan_x: float = 0.0
    initial_pan_y: float = 0.0
    initial_zoom: float = 1.0

    def to_payload(self) -> dict[str, Any]:
        """Convert the model into a plain JSON-serialisable structure."""
        return asdict(self)
