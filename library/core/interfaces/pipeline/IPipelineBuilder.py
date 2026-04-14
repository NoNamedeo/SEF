from __future__ import annotations

from abc import ABC, abstractmethod

# from library.core.artifacts.PipelineEvent import PipelineEvent
from library.core.pipeline.Pipeline import Pipeline


class IPipelineBuilder(ABC):
    """Builds a Pipeline from a PipelineEvent trigger."""

    @abstractmethod
    def build(self, event) -> Pipeline: ...
