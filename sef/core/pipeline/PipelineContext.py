from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from sef.core.interfaces.IFrameExporter import IFrameExporter
from sef.core.interfaces.IFrameExtractor import IFrameExtractor
from sef.core.interfaces.ISignalCleaner import ISignalCleaner
from sef.core.interfaces.ISignalExtractor import ISignalExtractor
from sef.core.interfaces.IVisualizer import IVisualizer
from sef.core.pipeline.IntermediateFrameCapture import IntermediateFrameCaptureConfig
from sef.core.pipeline.PipelineErrors import PipelineContextError
from sef.core.pipeline.StreamRuntimeConfig import StreamRuntimeConfig
from sef.core.pipeline.VisualizerBinding import VisualizerBinding


@dataclass(frozen=True)
class PipelineContext:
    """
    Immutable dependency graph for one pipeline execution unit.

    Design rationale
    ----------------
    `PipelineContext` owns construction invariants, not execution logic. It is
    a frozen value assembled by builders or factories before execution starts.
    `Pipeline` can therefore remain a thin facade over planning and runtime
    collaborators, while tests and application services can inspect a complete
    pipeline topology without running it.

    Lifecycle
    ---------
    A context is normally built once per submitted run. It may be reused when
    all contained component instances are themselves safe to reuse; components
    with per-run mutable state should be instantiated per run by the builder or
    registry factory.

    Thread safety
    -------------
    The context normalizes component collections to tuples and deep-copies
    `source_config`. It does not make contained component objects immutable or
    thread-safe.

    Attributes
    ----------
    frame_extractor:
        Required source component that produces frames.
    signal_extractor:
        Required component that turns processed frames into signal samples.
    analyzers:
        Non-empty analyzer sequence.
    frame_processors:
        Optional buffer-level preprocessing steps.
    frame_exporters:
        Optional final-output exporters that consume processed frames and
        return artifacts while preserving replayable frames.
    signal_cleaners:
        Optional signal smoothing, filtering, or normalization steps.
    visualizers:
        Visualizers applied to all analyzer results.
    visualizer_bindings:
        Selective visualizer-to-result bindings.
    intermediate_frame_capture:
        Bounded capture policy for frame-processing debug snapshots.
    intermediate_frame_visualizers:
        Visualizers dedicated to intermediate frame collections.
    stream_runtime:
        Bounded-buffer and latency-policy settings used by adaptive streaming.
    source_config:
        Compact construction metadata for reproducibility exports.

    Raises
    ------
    PipelineContextError
        If required components are missing, required sequences are empty, or a
        typed field receives an invalid object.
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
