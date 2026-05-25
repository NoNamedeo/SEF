from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from library.core.interfaces.IFrameExporter import IFrameExporter
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameCaptureConfig
from library.core.pipeline.PipelineErrors import PipelineContextError
from library.core.pipeline.StreamRuntimeConfig import StreamRuntimeConfig
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
    signal_extractor : converts processed frames into a trackable signal.
    analyzers        : at least one analyzer must be provided.

    Optional fields (default to empty collections)
    -----------------------------------------------
    frame_processors : zero or more buffer-level frame preprocessing steps.
    frame_exporters  : zero or more file-backed final-output exporters that
                       consume processed frames and return a replayable buffer.
    signal_cleaners  : zero or more smoothing / filtering steps on signals.
    visualizers      : zero or more rendering steps executed after analysis.
    visualizer_bindings
                     : optional selective visualizer-to-result mappings.
    intermediate_frame_capture
                     : optional bounded capture settings for frame processing
                       debug snapshots.
    intermediate_frame_visualizers
                     : optional visualizers that render the captured debug
                     collection, never normal analysis results.
    stream_runtime   : bounded-buffer and latency policy settings used by the
                       adaptive execution runtime.
    source_config    : optional construction metadata used by exporters to
                       recreate the original registry-driven configuration.
    """

    # ── Required (no default) ───────────────────────────────────────────────
    frame_extractor: IFrameExtractor
    signal_extractor: ISignalExtractor
    analyzers: Sequence[IAnalyzer]

    # ── Optional (with default) ─────────────────────────────────────────────
    frame_processors: Sequence[IFrameBufferProcessor] = field(default_factory=tuple)
    frame_exporters: Sequence[IFrameExporter] = field(default_factory=tuple)
    signal_cleaners: Sequence[ISignalCleaner] = field(default_factory=tuple)
    visualizers: Sequence[IVisualizer] = field(default_factory=tuple)
    visualizer_bindings: Sequence[VisualizerBinding] = field(default_factory=tuple)
    intermediate_frame_capture: IntermediateFrameCaptureConfig = field(
        default_factory=IntermediateFrameCaptureConfig.disabled
    )
    intermediate_frame_visualizers: Sequence[IVisualizer] = field(default_factory=tuple)
    stream_runtime: StreamRuntimeConfig = field(default_factory=StreamRuntimeConfig)
    source_config: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.frame_extractor is None:
            raise PipelineContextError("PipelineContext requires a frame_extractor.", path="frame_extractor")
        if self.signal_extractor is None:
            raise PipelineContextError("PipelineContext requires a signal_extractor.", path="signal_extractor")

        object.__setattr__(
            self,
            "analyzers",
            self._required_tuple("analyzers", self.analyzers),
        )
        object.__setattr__(
            self,
            "frame_processors",
            self._frame_processors_tuple(self.frame_processors),
        )
        object.__setattr__(
            self,
            "frame_exporters",
            self._frame_exporters_tuple(self.frame_exporters),
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
        if not isinstance(self.intermediate_frame_capture, IntermediateFrameCaptureConfig):
            raise PipelineContextError(
                "PipelineContext field 'intermediate_frame_capture' must be an IntermediateFrameCaptureConfig.",
                path="intermediate_frame_capture",
            )
        object.__setattr__(
            self,
            "intermediate_frame_visualizers",
            self._optional_tuple("intermediate_frame_visualizers", self.intermediate_frame_visualizers),
        )
        if not isinstance(self.stream_runtime, StreamRuntimeConfig):
            raise PipelineContextError(
                "PipelineContext field 'stream_runtime' must be a StreamRuntimeConfig.",
                path="stream_runtime",
            )
        self.stream_runtime.validate()
        object.__setattr__(self, "source_config", self._source_config_mapping(self.source_config))

    @staticmethod
    def _required_tuple(name: str, values: Sequence) -> tuple:
        items = PipelineContext._optional_tuple(name, values)
        if not items:
            raise PipelineContextError(f"PipelineContext requires at least one {name[:-1]}.", path=name)
        return items

    @staticmethod
    def _optional_tuple(name: str, values: Sequence | None) -> tuple:
        if values is None:
            raise PipelineContextError(f"PipelineContext field '{name}' cannot be None.", path=name)
        items = tuple(values)
        if any(item is None for item in items):
            raise PipelineContextError(f"PipelineContext field '{name}' cannot contain None.", path=name)
        return items

    @staticmethod
    def _visualizer_bindings_tuple(values: Sequence[VisualizerBinding] | None) -> tuple[VisualizerBinding, ...]:
        items = PipelineContext._optional_tuple("visualizer_bindings", values)
        if any(not isinstance(item, VisualizerBinding) for item in items):
            raise PipelineContextError(
                "PipelineContext field 'visualizer_bindings' must contain VisualizerBinding instances.",
                path="visualizer_bindings",
            )
        return items

    @staticmethod
    def _frame_processors_tuple(
        values: Sequence[IFrameBufferProcessor] | None,
    ) -> tuple[IFrameBufferProcessor, ...]:
        items = PipelineContext._optional_tuple("frame_processors", values)
        if any(not isinstance(item, IFrameBufferProcessor) for item in items):
            raise PipelineContextError(
                "PipelineContext field 'frame_processors' must contain IFrameBufferProcessor instances.",
                path="frame_processors",
            )
        return items

    @staticmethod
    def _frame_exporters_tuple(
        values: Sequence[IFrameExporter] | None,
    ) -> tuple[IFrameExporter, ...]:
        items = PipelineContext._optional_tuple("frame_exporters", values)
        if any(not isinstance(item, IFrameExporter) for item in items):
            raise PipelineContextError(
                "PipelineContext field 'frame_exporters' must contain IFrameExporter instances.",
                path="frame_exporters",
            )
        return items

    @staticmethod
    def _source_config_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise PipelineContextError("PipelineContext field 'source_config' must be a mapping.", path="source_config")
        return deepcopy(dict(value))
