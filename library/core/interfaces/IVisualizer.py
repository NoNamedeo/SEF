from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from library.core.interfaces.IData import IData
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext


class IVisualizer(ABC):
    capabilities = StageCapabilities.batch()

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        """Build UI-agnostic visual artifacts from analytical data."""
