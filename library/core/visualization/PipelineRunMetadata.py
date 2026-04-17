from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineRunMetadata:
    """Execution metadata attached to completed pipeline outputs."""

    pipeline_id: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pipeline_id:
            raise ValueError("PipelineRunMetadata.pipeline_id must be a non-empty string.")
        object.__setattr__(self, "execution_metadata", dict(self.execution_metadata))
