from __future__ import annotations

import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TextIO

from sef.cli.output import renderer_for
from sef.core.errors import PipelineConfigurationError, PipelineExecutionError


@dataclass(slots=True)
class DiagnosticItem:
    """Structured CLI diagnostic item rendered by `CliDiagnostics`."""

    severity: str
    message: str
    cause: str | None = None
    suggestion: str | None = None


@dataclass(slots=True)
class CliDiagnostics:
    """Collects human-readable CLI diagnostics and maps them to exit codes."""

    items: list[DiagnosticItem] = field(default_factory=list)
    exception: BaseException | None = None

    @classmethod
    def from_exception(cls, exc: BaseException) -> CliDiagnostics:
        """Build a diagnostic payload for an unexpected command failure."""
        if isinstance(exc, FileNotFoundError):
            return cls(
                [
                    DiagnosticItem(
                        "error",
                        str(exc),
                        cause="The requested path does not exist.",
                        suggestion="Check the path or run `sef init` to create a starter project.",
                    )
                ],
                exception=exc,
            )

        if isinstance(exc, PermissionError):
            return cls(
                [
                    DiagnosticItem(
                        "error",
                        str(exc),
                        cause="The CLI cannot read or write a required path.",
                        suggestion="Check directory permissions or choose a writable --output directory.",
                    )
                ],
                exception=exc,
            )

        if isinstance(exc, PipelineConfigurationError):
            return cls(
                [
                    DiagnosticItem(
                        "error",
                        str(exc),
                        cause="The pipeline config could not be resolved into executable components.",
                        suggestion="Run `sef validate <config> --strict` and inspect component names/params.",
                    )
                ],
                exception=exc,
            )

        if isinstance(exc, PipelineExecutionError):
            return cls(
                [
                    DiagnosticItem(
                        "error",
                        str(exc),
                        cause="The pipeline started but a runtime stage failed.",
                        suggestion="Re-run with `--debug` for a traceback and inspect the failing component.",
                    )
                ],
                exception=exc,
            )

        return cls(
            [
                DiagnosticItem(
                    "error",
                    str(exc) or type(exc).__name__,
                    cause="The CLI command failed before completion.",
                    suggestion="Re-run with `--debug` to include the full traceback.",
                )
            ],
            exception=exc,
        )

    def add_warning(self, message: str, *, cause: str | None = None, suggestion: str | None = None) -> None:
        """Append a non-blocking diagnostic."""
        self.items.append(DiagnosticItem("warning", message, cause, suggestion))

    def add_error(self, message: str, *, cause: str | None = None, suggestion: str | None = None) -> None:
        """Append a blocking diagnostic."""
        self.items.append(DiagnosticItem("error", message, cause, suggestion))

    @property
    def has_errors(self) -> bool:
        """Return True when at least one item should fail the command."""
        return any(item.severity == "error" for item in self.items)

    def extend(self, items: Sequence[DiagnosticItem]) -> None:
        """Append diagnostics from another command layer."""
        self.items.extend(items)

    def print(self, *, debug: bool = False, stream: TextIO | None = None) -> None:
        """Render collected diagnostics."""
        target = stream or sys.stderr
        renderer = renderer_for(target)
        for item in self.items:
            label = "error" if item.severity == "error" else "warning"
            renderer.print_status(label, item.message, stream=target)
            if item.cause:
                renderer.print_detail("cause", item.cause, stream=target)
            if item.suggestion:
                renderer.print_detail("suggestion", item.suggestion, stream=target)
        if debug and self.exception is not None:
            print("", file=target)
            traceback.print_exception(self.exception, file=target)

    def exit_code(self) -> int:
        """Return the CLI exit code represented by these diagnostics."""
        return 1 if self.has_errors else 0
