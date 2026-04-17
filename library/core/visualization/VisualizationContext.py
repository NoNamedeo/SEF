from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualizationContext:
    """Execution context passed to visualizers during artifact generation."""

    pipeline_id: str | None = None
    analyzer_name: str | None = None
    visualizer_name: str | None = None
    result_index: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    execution_metadata: Mapping[str, Any] = field(default_factory=dict)
    render_hints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))
        object.__setattr__(self, "execution_metadata", dict(self.execution_metadata))
        object.__setattr__(self, "render_hints", dict(self.render_hints))
