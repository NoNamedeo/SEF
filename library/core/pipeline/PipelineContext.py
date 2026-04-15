from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.pipeline.VisualizerBinding import VisualizerBinding


@dataclass(frozen=True)
class PipelineContext:
    """
    Pure dependency holder for the pipeline execution unit.

    Design rationale
    ----------------
    PipelineContext owns construction invariants, not execution logic. It is
    an immutable bag of collaborators resolved by a builder/factory before
    execution. This keeps Pipeline itself completely stateless with respect to
    construction decisions, and makes each context safely reusable and testable
    in isolation.

    Field ordering follows the dataclass rule: fields WITH defaults must
    come after fields WITHOUT defaults.

    Required fields
    ---------------
    frame_extractor  : entry-point of the pipeline; must always be present.
    signal_extractor : converts cleaned frames into a trackable signal.
    analyzers        : at least one analyzer must be provided.

    Optional fields (default to empty collections)
    -----------------------------------------------
    frame_cleaners   : zero or more pre-processing steps on raw frames.
    signal_cleaners  : zero or more smoothing / filtering steps on signals.
    visualizers      : zero or more rendering steps executed after analysis.
    visualizer_bindings
                     : optional selective visualizer-to-result mappings.
    """

    # ── Required (no default) ───────────────────────────────────────────────
    frame_extractor: IFrameExtractor
    signal_extractor: ISignalExtractor
    analyzers: Sequence[IAnalyzer]

    # ── Optional (with default) ─────────────────────────────────────────────
    frame_cleaners: Sequence[IFrameCleaner] = field(default_factory=tuple)
    signal_cleaners: Sequence[ISignalCleaner] = field(default_factory=tuple)
    visualizers: Sequence[IVisualizer] = field(default_factory=tuple)
    visualizer_bindings: Sequence[VisualizerBinding] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.frame_extractor is None:
            raise ValueError("PipelineContext requires a frame_extractor.")
        if self.signal_extractor is None:
            raise ValueError("PipelineContext requires a signal_extractor.")

        object.__setattr__(
            self,
            "analyzers",
            self._required_tuple("analyzers", self.analyzers),
        )
        object.__setattr__(
            self,
            "frame_cleaners",
            self._optional_tuple("frame_cleaners", self.frame_cleaners),
        )
        object.__setattr__(
            self,
            "signal_cleaners",
            self._optional_tuple("signal_cleaners", self.signal_cleaners),
        )
        object.__setattr__(
            self,
            "visualizers",
            self._optional_tuple("visualizers", self.visualizers),
        )
        object.__setattr__(
            self,
            "visualizer_bindings",
            self._visualizer_bindings_tuple(self.visualizer_bindings),
        )

    @staticmethod
    def _required_tuple(name: str, values: Sequence) -> tuple:
        items = PipelineContext._optional_tuple(name, values)
        if not items:
            raise ValueError(f"PipelineContext requires at least one {name[:-1]}.")
        return items

    @staticmethod
    def _optional_tuple(name: str, values: Sequence | None) -> tuple:
        if values is None:
            raise ValueError(f"PipelineContext field '{name}' cannot be None.")
        items = tuple(values)
        if any(item is None for item in items):
            raise ValueError(f"PipelineContext field '{name}' cannot contain None.")
        return items

    @staticmethod
    def _visualizer_bindings_tuple(values: Sequence[VisualizerBinding] | None) -> tuple[VisualizerBinding, ...]:
        items = PipelineContext._optional_tuple("visualizer_bindings", values)
        if any(not isinstance(item, VisualizerBinding) for item in items):
            raise ValueError("PipelineContext field 'visualizer_bindings' must contain VisualizerBinding instances.")
        return items
