from __future__ import annotations

from dataclasses import dataclass, field

from sef.core.artifacts.intermediate_frame.IntermediateFrameArtifacts import IntermediateFrameArtifactCollection
from sef.core.interfaces.IData import IData
from sef.core.visualization.PipelineRunMetadata import PipelineRunMetadata
from sef.core.visualization.VisualArtifact import VisualArtifact


@dataclass(frozen=True, slots=True)
class PipelineOutputs:
    """
    Immutable aggregate returned by completed pipeline runs.

    `PipelineOutputs` is the handoff value for UI, API, notebook, and exporter
    adapters. It keeps analytical results separate from final artifacts and
    debug artifacts so presentation layers can decide what to show by default.

    Attributes
    ----------
    results:
        Final analyzer results in analyzer order.
    final_artifacts:
        Primary artifacts intended for normal users.
    debug_artifacts:
        Diagnostic artifacts intended for inspection or troubleshooting.
    metadata:
        Execution metadata, plan, and reproducibility details.
    intermediate_frames:
        Optional bounded intermediate-frame debug collection.
    """

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
