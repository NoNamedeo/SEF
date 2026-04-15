from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from library.core.interfaces.IVisualizer import IVisualizer


@dataclass(frozen=True, slots=True)
class VisualizerBinding:
    """
    Binds a visualizer to selected analyzer result indexes.

    ``result_indices=None`` preserves the default behaviour: the visualizer is
    applied to every analyzer result. Providing indexes makes the binding
    selective without changing the visualizer contract.
    """

    visualizer: IVisualizer
    result_indices: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.visualizer is None:
            raise ValueError("VisualizerBinding requires a visualizer.")
        if self.result_indices is None:
            return
        indices = tuple(self.result_indices)
        if any(not isinstance(index, int) for index in indices):
            raise ValueError("VisualizerBinding result_indices must contain integers.")
        if any(index < 0 for index in indices):
            raise ValueError("VisualizerBinding result_indices cannot contain negative indexes.")
        object.__setattr__(self, "result_indices", indices)
