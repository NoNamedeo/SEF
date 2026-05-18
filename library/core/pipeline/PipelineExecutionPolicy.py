from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from library.core.pipeline.PipelineContext import PipelineContext


class PipelineExecutionMode(str, Enum):
    """Supported execution modes for one pipeline stage."""

    BATCH = "batch"
    STREAMING = "streaming"


@dataclass(frozen=True, slots=True)
class PipelineExecutionDecision:
    """
    Policy decision for one stage.

    The decision carries both the selected mode and the reason that will appear
    in execution plans. Keeping the explanation in the policy prevents planner
    and runtime from inventing separate narratives for the same choice.
    """

    mode: PipelineExecutionMode
    reason: str

    @property
    def streams(self) -> bool:
        """Return True when this decision selects streaming execution."""
        return self.mode == PipelineExecutionMode.STREAMING


@dataclass(frozen=True, slots=True)
class PipelineExecutionEstimates:
    """
    Lightweight size estimates used by interchangeable execution policies.

    Estimates are deliberately optional. A policy may use them to make
    cost-aware choices, while still falling back to capability and downstream
    information when dimensions are unknown.
    """

    frame_queue_bytes: int | None
    materialized_frame_bytes: int | None
    signal_queue_bytes: int
    data_queue_bytes: int

    @classmethod
    def from_context(cls, context: PipelineContext) -> PipelineExecutionEstimates:
        frame_bytes = cls._estimated_frame_bytes(context)
        return cls(
            frame_queue_bytes=cls._queue_bytes(
                frame_bytes,
                context.stream_runtime.frame_buffer_size,
            ),
            materialized_frame_bytes=cls._materialized_bytes(frame_bytes, context),
            signal_queue_bytes=context.stream_runtime.signal_buffer_size * 1024,
            data_queue_bytes=context.stream_runtime.data_buffer_size * 1024,
        )

    @staticmethod
    def _estimated_frame_bytes(context: PipelineContext) -> int | None:
        resize = getattr(context.frame_extractor, "resize", None)
        if not isinstance(resize, (tuple, list)) or len(resize) != 2:
            return None
        width, height = int(resize[0]), int(resize[1])
        if width <= 0 or height <= 0:
            return None
        return width * height * 3

    @staticmethod
    def _queue_bytes(frame_bytes: int | None, capacity: int) -> int | None:
        if frame_bytes is None:
            return None
        return frame_bytes * capacity

    @staticmethod
    def _materialized_bytes(frame_bytes: int | None, context: PipelineContext) -> int | None:
        if frame_bytes is None:
            return None
        max_frames = getattr(context.frame_extractor, "max_frames", None)
        if max_frames is None:
            return None
        return frame_bytes * int(max_frames)


@dataclass(frozen=True, slots=True)
class PipelineStagePolicyContext:
    """
    Inputs available to execution-mode policies for one stage.

    The context contains facts, not decisions: capabilities, current stream
    state, downstream demand and optional cost estimates. This keeps policies
    replaceable and testable without depending on executor internals.
    """

    stage_id: str
    stage_group: str
    stage_streamable: bool
    input_is_streaming: bool = False
    downstream_streamable: bool = False
    progressive_consumer: bool = False
    estimated_queue_bytes: int | None = None
    estimated_materialized_bytes: int | None = None


class PipelineExecutionPolicy(Protocol):
    """
    Strategy interface for batch/streaming execution decisions.

    Custom policies can implement latency-first, memory-first, observability or
    domain-specific rules without changing the execution planner or runtime.
    Implementations must never select streaming when ``stage_streamable`` is
    false.
    """

    def decide_source(self, context: PipelineStagePolicyContext) -> PipelineExecutionDecision:
        """Choose how a source stage should produce its output."""

    def decide_stage(self, context: PipelineStagePolicyContext) -> PipelineExecutionDecision:
        """Choose how a normal transformation/export stage should run."""

    def decide_analyzer(self, context: PipelineStagePolicyContext) -> PipelineExecutionDecision:
        """Choose how an analyzer should consume signal input."""


class DefaultPipelineExecutionPolicy:
    """
    Default cost-aware streaming policy.

    The policy keeps streaming when it avoids materializing an active stream,
    starts new streaming segments only when there is downstream demand, and uses
    available size estimates to prefer bounded queues over full materialization
    when that is clearly cheaper.
    """

    def decide_source(self, context: PipelineStagePolicyContext) -> PipelineExecutionDecision:
        if not context.stage_streamable:
            return self._batch("source exposes only a batch contract")
        if not self._has_streaming_demand(context):
            return self._batch("batch source avoids a stream with no progressive consumer")
        return self._stream("source opens a streaming segment for downstream demand")

    def decide_stage(self, context: PipelineStagePolicyContext) -> PipelineExecutionDecision:
        if not context.stage_streamable:
            return self._batch("stage requires a complete input sequence")
        if context.input_is_streaming:
            return self._stream("preserves active stream and avoids materialization")
        if context.progressive_consumer:
            return self._stream("opens streaming segment for progressive consumer")
        if not context.downstream_streamable:
            return self._batch("batch mode avoids an isolated streaming switch")
        if self._bounded_stream_reduces_memory(context):
            return self._stream("bounded stream is cheaper than estimated materialization")
        return self._stream("opens streaming segment for downstream streamable stage")

    def decide_analyzer(self, context: PipelineStagePolicyContext) -> PipelineExecutionDecision:
        if not context.stage_streamable:
            return self._batch("analyzer requires a complete signal")
        if context.input_is_streaming:
            return self._stream("consumes active signal stream")
        if context.progressive_consumer:
            return self._stream("publishes progressive data for streaming visualizer")
        return self._batch("batch analyzer avoids isolated progressive execution")

    @staticmethod
    def _has_streaming_demand(context: PipelineStagePolicyContext) -> bool:
        return context.downstream_streamable or context.progressive_consumer

    @staticmethod
    def _bounded_stream_reduces_memory(context: PipelineStagePolicyContext) -> bool:
        queue_bytes = context.estimated_queue_bytes
        materialized_bytes = context.estimated_materialized_bytes
        return (
            queue_bytes is not None
            and materialized_bytes is not None
            and queue_bytes < materialized_bytes
        )

    @staticmethod
    def _batch(reason: str) -> PipelineExecutionDecision:
        return PipelineExecutionDecision(PipelineExecutionMode.BATCH, reason)

    @staticmethod
    def _stream(reason: str) -> PipelineExecutionDecision:
        return PipelineExecutionDecision(PipelineExecutionMode.STREAMING, reason)
