from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PipelineDiagnosticsLevel(str, Enum):
    """Amount of execution diagnostics attached to completed pipeline outputs."""

    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class PipelineRunOptions:
    """
    Controls optional run metadata that is not required to execute a pipeline.

    The default is intentionally lightweight: execution plans and
    reproducibility exports are omitted unless the caller explicitly requests
    them. The public ``Pipeline.execution_plan()`` method remains available
    independently from these options.
    """

    diagnostics: PipelineDiagnosticsLevel | str = PipelineDiagnosticsLevel.NONE
    reproducibility: bool = False

    def __post_init__(self) -> None:
        try:
            diagnostics = PipelineDiagnosticsLevel(self.diagnostics)
        except ValueError as exc:
            allowed = ", ".join(level.value for level in PipelineDiagnosticsLevel)
            raise ValueError(f"diagnostics must be one of: {allowed}.") from exc
        object.__setattr__(self, "diagnostics", diagnostics)

        if not isinstance(self.reproducibility, bool):
            raise TypeError("reproducibility must be a boolean.")

    @property
    def includes_execution_plan(self) -> bool:
        """Return whether the run must build an execution plan."""
        return self.diagnostics is not PipelineDiagnosticsLevel.NONE

    @classmethod
    def lightweight(cls) -> PipelineRunOptions:
        """Return options for the lowest-overhead execution path."""
        return cls()

    @classmethod
    def full(cls) -> PipelineRunOptions:
        """Return options that preserve complete diagnostics and exports."""
        return cls(
            diagnostics=PipelineDiagnosticsLevel.FULL,
            reproducibility=True,
        )
