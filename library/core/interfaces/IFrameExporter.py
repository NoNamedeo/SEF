from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.visualization.VisualArtifact import VisualArtifact


@dataclass(frozen=True, slots=True)
class FrameExportContext:
    """Runtime metadata passed to frame-buffer exporters."""

    pipeline_id: str | None
    exporter_name: str
    execution_metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FrameExportResult:
    """Result of an exporter that preserves the frame stream for downstream stages."""

    buffer: FrameBuffer
    artifacts: tuple[VisualArtifact, ...]


class IFrameExporter(ABC):
    """
    Export final artifacts from processed frames without being a frame processor.

    Exporters may write file-backed artifacts and must return a buffer that can
    still be consumed by signal extractors.
    """

    capabilities = StageCapabilities.batch(stateful=False)

    @abstractmethod
    def export(self, buffer: FrameBuffer, context: FrameExportContext) -> FrameExportResult:
        """Persist or build artifacts from the frame stream and return the next buffer."""
