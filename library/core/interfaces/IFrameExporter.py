from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.visualization.VisualArtifact import VisualArtifact


@dataclass(frozen=True, slots=True)
class FrameExportContext:
    """
    Runtime metadata passed to frame-buffer exporters.

    Attributes
    ----------
    pipeline_id:
        Optional run id.
    exporter_name:
        Human-readable exporter component name.
    execution_metadata:
        Metadata supplied by the caller and propagated through the run.
    """

    pipeline_id: str | None
    exporter_name: str
    execution_metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FrameExportResult:
    """
    Exporter result that preserves frames for downstream stages.

    Exporters may create artifacts but must also return a frame buffer so signal
    extraction can continue after export.
    """

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
        """
        Persist or build artifacts from the frame stream.

        Parameters
        ----------
        buffer:
            Processed frame stream.
        context:
            Runtime metadata for artifact naming and reproducibility.

        Returns
        -------
        FrameExportResult
            Forwarded frame buffer plus generated artifacts.
        """
