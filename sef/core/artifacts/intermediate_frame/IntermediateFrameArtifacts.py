from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sef.core.artifacts.mask.MaskArtifacts import IntermediateFrameArtifact
from sef.core.interfaces.IData import IData


class IntermediateFrameArtifactExporter(Protocol):
    """Port implemented by infrastructure exporters for debug frame artifacts."""

    def export_many(self, artifacts: tuple[IntermediateFrameArtifact, ...]) -> tuple[Path, ...]:
        """Export artifacts and return written paths."""


IntermediateFrameArtifactExporterFactory = Callable[[Path], IntermediateFrameArtifactExporter]

_exporter_factory: IntermediateFrameArtifactExporterFactory | None = None


def set_intermediate_frame_exporter_factory(factory: IntermediateFrameArtifactExporterFactory | None) -> None:
    """
    Register the process-local exporter factory used by ``export()``.

    The core artifact model owns the collection and metadata. Concrete image
    writing is infrastructure behavior, so builtin packages or applications
    provide the exporter implementation through this hook.
    """
    global _exporter_factory
    _exporter_factory = factory


def create_intermediate_frame_exporter(output_directory: Path | str) -> IntermediateFrameArtifactExporter:
    """Create the registered intermediate-frame exporter."""
    if _exporter_factory is None:
        raise RuntimeError(
            "Intermediate frame export requires an exporter factory. "
            "Import sef.builtin or register one with set_intermediate_frame_exporter_factory()."
        )
    return _exporter_factory(Path(output_directory))


@dataclass(frozen=True, slots=True)
class IntermediateFrameArtifactCollection(IData):
    """
    Immutable debug stream produced by frame processing stages.

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
        return create_intermediate_frame_exporter(Path(target)).export_many(self.artifacts)
