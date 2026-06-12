from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.interfaces.IData import IData
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext


class IVisualizer(ABC):
    """
    Batch contract for converting analytical data into visual artifacts.

    Visualizers must return UI-agnostic `VisualArtifact` values. They should not
    require Streamlit containers, OpenCV windows, notebook globals, or web
    framework state unless they are explicitly adapter implementations outside
    the core package.
    """

    capabilities = StageCapabilities.batch()

    def __init__(self, config: dict[str, Any] | None = None):
        """Store plugin-specific visualizer configuration."""
        self.config = config or {}

    @abstractmethod
    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        """
        Build visual artifacts from analytical data.

        Parameters
        ----------
        data:
            Analyzer result to render.
        context:
            Optional run metadata, result index, and rendering hints.

        Returns
        -------
        tuple[VisualArtifact, ...]
            Final or debug artifacts for UI and exporter adapters.
        """
