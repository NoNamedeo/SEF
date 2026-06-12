from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from sef.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore


@dataclass(frozen=True, slots=True)
class FrameProcessorExecutionContext:
    """Runtime context offered to processors that support debug capture."""

    processor_index: int
    processor_name: str
    stage_name: str
    intermediate_store: IntermediateFrameArtifactStore | None


class ContextAwareFrameProcessor(Protocol):
    """Optional processor protocol for capture-aware frame processing."""

    def process_with_context(
        self,
        buffer: FrameBuffer,
        context: FrameProcessorExecutionContext,
    ) -> FrameBuffer:
        """Process a buffer with access to pipeline execution context."""


class FrameProcessingStage:
    """
    Apply ordered buffer-level frame processors.

    The stage is intentionally buffer-oriented: simple single-frame processors
    are adapted before they reach this boundary, while temporal processors can
    work on the complete sequence without hidden state.
    """

    def apply(
        self,
        buffer: FrameBuffer,
        frame_processors: Sequence[IFrameBufferProcessor],
        intermediate_store: IntermediateFrameArtifactStore | None = None,
    ) -> FrameBuffer:
        if not frame_processors:
            return buffer

        processed_buffer = buffer
        for processor_index, processor in enumerate(frame_processors):
            processed_buffer = self._process(
                processed_buffer,
                processor,
                processor_index=processor_index,
                intermediate_store=intermediate_store,
            )
        return processed_buffer

    @staticmethod
    def _process(
        buffer: FrameBuffer,
        processor: IFrameBufferProcessor,
        *,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore | None,
    ) -> FrameBuffer:
        context = FrameProcessorExecutionContext(
            processor_index=processor_index,
            processor_name=type(processor).__name__,
            stage_name=f"frame_processing[{processor_index}].{type(processor).__name__}",
            intermediate_store=intermediate_store,
        )
        process_with_context = getattr(processor, "process_with_context", None)
        if callable(process_with_context):
            return process_with_context(buffer, context)
        return processor.process(buffer)
