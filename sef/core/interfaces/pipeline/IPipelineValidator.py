from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.core.pipeline.Pipeline import Pipeline


class IPipelineValidator(ABC):
    """
    Validation port for executable pipeline instances.

    Validators are application-level checks. They should not mutate the
    pipeline and should raise typed configuration or validation errors when a
    run cannot be accepted.
    """

    @abstractmethod
    def validate(self, pipeline: "Pipeline") -> None:
        """Validate a pipeline before execution."""
        pass
