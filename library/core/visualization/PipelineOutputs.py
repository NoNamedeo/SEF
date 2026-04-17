from __future__ import annotations

from dataclasses import dataclass

from library.core.interfaces.IData import IData
from library.core.visualization.PipelineRunMetadata import PipelineRunMetadata
from library.core.visualization.VisualArtifact import VisualArtifact


@dataclass(frozen=True, slots=True)
class PipelineOutputs:
    """Immutable aggregate of analysis results and presentation artifacts."""

    results: tuple[IData, ...]
    artifacts: tuple[VisualArtifact, ...]
    metadata: PipelineRunMetadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
