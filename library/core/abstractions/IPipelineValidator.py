from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.core.pipeline.Pipeline import Pipeline

class IPipelineValidator(ABC):

    @abstractmethod
    def validate(self, pipeline: "Pipeline") -> None:
        """Validate a pipeline before execution."""
        pass
