from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TextIO


_NO_COLOR_ENV = "NO_COLOR"


@dataclass(frozen=True, slots=True)
class CliPalette:
    """ANSI color palette used by the SEF CLI presentation layer."""

    reset: str = "\033[0m"
    bold: str = "\033[1m"
    dim: str = "\033[2m"
    red: str = "\033[31m"
    green: str = "\033[32m"
    yellow: str = "\033[33m"
    cyan: str = "\033[36m"


class CliMessageRenderer:
    """
    Render branded CLI messages without adding external dependencies.

    The renderer is intentionally presentation-only: command handlers still
    produce plain strings and structured diagnostics; this class decides how to
    display them in a terminal.
    """

    _LABEL_COLORS = {
        "ok": "green",
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
    }

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        enable_color: bool | None = None,
        brand: str = "SEF",
        palette: CliPalette | None = None,
    ) -> None:
        self._stream = stream or sys.stdout
        self._brand = brand
        self._palette = palette or CliPalette()
        self._enable_color = self._should_enable_color(enable_color)

    def status(self, level: str, message: str) -> str:
        """Return one branded status line."""
        normalized = level.lower().strip()
        return f"{self.brand()}  {self.label(normalized):<14} {message}"

    def detail(self, name: str, value: str) -> str:
        """Return an indented diagnostic detail line."""
        prefix = self._style(f"{name}:", self._palette.dim)
        return f"  {prefix} {value}"

    def brand(self) -> str:
        """Return the styled framework name."""
        return self._style(self._brand, self._palette.bold + self._palette.cyan)

    def label(self, level: str) -> str:
        """Return a styled severity/status label."""
        color_name = self._LABEL_COLORS.get(level, "cyan")
        color = getattr(self._palette, color_name)
        return self._style(level, color)

    def print_status(self, level: str, message: str, *, stream: TextIO | None = None) -> None:
        """Print one branded status line."""
        print(self.status(level, message), file=stream or self._stream)

    def print_detail(self, name: str, value: str, *, stream: TextIO | None = None) -> None:
        """Print one indented diagnostic detail line."""
        print(self.detail(name, value), file=stream or self._stream)

    def _style(self, text: str, ansi: str) -> str:
        if not self._enable_color:
            return text
        return f"{ansi}{text}{self._palette.reset}"

    def _should_enable_color(self, override: bool | None) -> bool:
        if override is not None:
            return override
        if os.environ.get(_NO_COLOR_ENV) is not None:
            return False
        return bool(getattr(self._stream, "isatty", lambda: False)())


def renderer_for(stream: TextIO | None = None, *, enable_color: bool | None = None) -> CliMessageRenderer:
    """Create a renderer for a specific output stream."""
    return CliMessageRenderer(stream=stream, enable_color=enable_color)
