from __future__ import annotations


from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from library.core.artifacts.MaskArtifacts import IntermediateFrameArtifact
from library.exporters.IntermediateFrameArtifactExporter import IntermediateFrameArtifactExporter
from library.core.interfaces.IData import IData


@dataclass(frozen=True, slots=True)
class IntermediateFrameArtifactCollection(IData):
    """
    Immutable debug stream produced by frame cleaning stages.

    The collection is deliberately separate from analysis results so existing
    visualizers keep receiving only the result types they already support.
    """

    artifacts: tuple[IntermediateFrameArtifact, ...] = ()
    title: str = "Intermediate Frame Snapshots"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        artifacts = tuple(self.artifacts)
        if any(not isinstance(artifact, IntermediateFrameArtifact) for artifact in artifacts):
            raise TypeError("IntermediateFrameArtifactCollection.artifacts must contain IntermediateFrameArtifact instances.")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def empty(cls) -> IntermediateFrameArtifactCollection:
        """Return an empty collection for pipelines without debug capture."""
        return cls()

    @property
    def count(self) -> int:
        """Return the number of stored intermediate snapshots."""
        return len(self.artifacts)

    @property
    def is_empty(self) -> bool:
        """Return whether no intermediate snapshots were captured."""
        return not self.artifacts

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return stage names in first-seen order."""
        return tuple(dict.fromkeys(artifact.stage_name for artifact in self.artifacts))

    @property
    def frame_indices(self) -> tuple[int | None, ...]:
        """Return frame indexes in first-seen order."""
        return tuple(dict.fromkeys(artifact.frame_index for artifact in self.artifacts))

    def by_stage_name(self, stage_name: str) -> tuple[IntermediateFrameArtifact, ...]:
        """Return snapshots captured for a specific stage."""
        return tuple(artifact for artifact in self.artifacts if artifact.stage_name == stage_name)

    def by_frame_index(self, frame_index: int | None) -> tuple[IntermediateFrameArtifact, ...]:
        """Return snapshots captured for a specific frame index."""
        return tuple(artifact for artifact in self.artifacts if artifact.frame_index == frame_index)

    def export(self, output_directory: Path | str | None = None) -> tuple[Path, ...]:
        """
        Persist stored frames to PNG files and return all paths written.

        If *output_directory* is omitted, the collection must have an
        ``export_directory`` entry in metadata. This supports deferred saving:
        the pipeline can capture bounded snapshots now, while developers decide
        later whether to write them to disk.
        """
        target = output_directory or self.metadata.get("export_directory")
        if target is None:
            raise ValueError("Intermediate frame export requires an output directory.")
        return IntermediateFrameArtifactExporter(Path(target)).export_many(self.artifacts)
