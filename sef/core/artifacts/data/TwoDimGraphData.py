from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sef.core.interfaces.IData import IData


@dataclass(slots=True)
class TwoDimGraphData(IData):
    """Chart-ready series returned by analyzers."""

    x: list[float]
    y: list[float]
    label: str = "signal"
    title: str = "Signal Analysis"
    x_label: str = "X"
    y_label: str = "Y"
    metadata: dict[str, Any] = field(default_factory=dict)
