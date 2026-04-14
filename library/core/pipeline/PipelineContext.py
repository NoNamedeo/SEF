from __future__ import annotations

from dataclasses import dataclass, field

from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.IVisualizer import IVisualizer


@dataclass(frozen=True)
class PipelineContext:
    """
    Pure dependency holder for the pipeline execution unit.

    Design rationale
    ----------------
    PipelineContext owns NO logic — it is an immutable bag of
    collaborators resolved by the builder/orchestrator before execution.
    This keeps Pipeline itself completely stateless with respect to
    construction decisions, and makes each context trivially serialisable,
    cloneable and testable in isolation.

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
    """

    # ── Required (no default) ───────────────────────────────────────────────
    frame_extractor: IFrameExtractor
    signal_extractor: ISignalExtractor
    analyzers: list[IAnalyzer]

    # ── Optional (with default) ─────────────────────────────────────────────
    frame_cleaners: list[IFrameCleaner] = field(default_factory=list)
    signal_cleaners: list[ISignalCleaner] = field(default_factory=list)
    visualizers: list[IVisualizer] = field(default_factory=list)
