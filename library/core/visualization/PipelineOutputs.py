from __future__ import annotations

from dataclasses import dataclass, field

from library.core.artifacts.IntermediateFrameArtifacts import IntermediateFrameArtifactCollection
from library.core.interfaces.IData import IData
from library.core.visualization.PipelineRunMetadata import PipelineRunMetadata
from library.core.visualization.VisualArtifact import VisualArtifact


@dataclass(frozen=True, slots=True)
class PipelineOutputs:
    """Immutable aggregate of analytical results, final artifacts, and debug artifacts."""

    results: tuple[IData, ...]
    final_artifacts: tuple[VisualArtifact, ...]
    debug_artifacts: tuple[VisualArtifact, ...]
    metadata: PipelineRunMetadata
    intermediate_frames: IntermediateFrameArtifactCollection = field(
        default_factory=IntermediateFrameArtifactCollection.empty
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "final_artifacts", tuple(self.final_artifacts))
        object.__setattr__(self, "debug_artifacts", tuple(self.debug_artifacts))
        if not isinstance(self.intermediate_frames, IntermediateFrameArtifactCollection):
            raise TypeError("PipelineOutputs.intermediate_frames must be an IntermediateFrameArtifactCollection.")

    @property
    def artifact_count(self) -> int:
        """Return final plus debug artifact count for monitoring summaries."""
        return len(self.final_artifacts) + len(self.debug_artifacts)
