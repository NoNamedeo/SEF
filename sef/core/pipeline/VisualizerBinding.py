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

    def target_indexes(self, result_count: int) -> tuple[int, ...]:
        """
        Resolve this binding against the available analyzer result indexes.

        The method belongs to the value object because index validation is part
        of the binding contract, not a concern of planner or renderer code.
        """
        if result_count < 0:
            raise ValueError("VisualizerBinding result_count cannot be negative.")
        if self.result_indices is None:
            return tuple(range(result_count))

        invalid = [index for index in self.result_indices if index >= result_count]
        if invalid:
            available = f"0..{result_count - 1}" if result_count > 0 else "<none>"
            raise ValueError(
                f"Visualizer target index out of range: {invalid}; "
                f"available result indexes: {available}."
            )
        return tuple(self.result_indices)
