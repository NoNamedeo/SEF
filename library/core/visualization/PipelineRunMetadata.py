from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineRunMetadata:
    """
    Execution metadata attached to completed pipeline outputs.

    Metadata is intended for reproducibility, UI summaries, and integration
    diagnostics. Mappings are copied to dictionaries so consumers can inspect a
    stable snapshot of the completed run.
    """

    pipeline_id: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_metadata: Mapping[str, Any] = field(default_factory=dict)
    execution_plan: Mapping[str, Any] = field(default_factory=dict)
    reproducibility: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pipeline_id:
            raise ValueError("PipelineRunMetadata.pipeline_id must be a non-empty string.")
        object.__setattr__(self, "execution_metadata", dict(self.execution_metadata))
        object.__setattr__(self, "execution_plan", dict(self.execution_plan))
        object.__setattr__(self, "reproducibility", dict(self.reproducibility))
