from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from library.core.interfaces.IData import IData


@dataclass(slots=True)
class VectorFieldGraphData(IData):
    """
    Data container for vector field (quiver) visualization.
    (x, y) define the position of each vector
    (u, v) define the vector components
    """

    x: list[float]
    y: list[float]
    u: list[float]
    v: list[float]

    title: str = "Vector Field"
    x_label: str = "X"
    y_label: str = "Y"

    metadata: dict[str, Any] = field(default_factory=dict)
