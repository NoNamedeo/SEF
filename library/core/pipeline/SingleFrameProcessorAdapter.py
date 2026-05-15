from __future__ import annotations

from typing import Any

from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.MaskArtifacts import IntermediateFrameArtifact
from library.core.interfaces.IFrameBufferProcessor import FrameProcessorCapabilities, IFrameBufferProcessor
from library.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from library.core.pipeline.FrameProcessingStage import FrameProcessorExecutionContext
from library.core.pipeline.IntermediateFrameCapture import (
    IntermediateFrameArtifactStore,
    IntermediateFrameCaptureContext,
)


class SingleFrameProcessorAdapter(IFrameBufferProcessor):
    """
    Adapt an ISingleFrameProcessor to the buffer-level processing pipeline.

    This keeps simple frame transformations small and testable while allowing
    the preprocessing stage to operate uniformly on FrameBuffer instances.
    """

    capabilities = FrameProcessorCapabilities(
        supports_streaming=True,
        requires_complete_sequence=False,
    )

    def __init__(self, single_frame_processor: ISingleFrameProcessor, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.single_frame_processor = single_frame_processor

    def process(self, buffer: FrameBuffer) -> FrameBuffer:
        """Apply the wrapped single-frame processor without debug capture."""
        return self._process(buffer, processor_index=0, intermediate_store=None)

    def process_with_context(
        self,
        buffer: FrameBuffer,
        context: FrameProcessorExecutionContext,
    ) -> FrameBuffer:
        """Apply the wrapped single-frame processor and emit snapshots when enabled."""
        return self._process(
            buffer,
            processor_index=context.processor_index,
            intermediate_store=context.intermediate_store,
        )

    def process_into(
        self,
        input_buffer: FrameBuffer,
        output_buffer: FrameBuffer,
        *,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore | None,
    ) -> None:
        """
        Stream processed frames from ``input_buffer`` into ``output_buffer``.

        This method is used by the pipeline runtime when the wrapped processor
        is stateless and can be applied one frame at a time without materializing
        the full intermediate sequence.
        """
        try:
            for source_sequence_index, frame in enumerate(input_buffer):
                output_buffer.put(
                    self._process_frame(
                        frame,
                        processor_index=processor_index,
                        source_sequence_index=source_sequence_index,
                        intermediate_store=intermediate_store,
                    )
                )
        finally:
            output_buffer.close()

    def _process(
        self,
        buffer: FrameBuffer,
        *,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore | None,
    ) -> FrameBuffer:
        processed_buffer = buffer.clone_empty()
        for source_sequence_index, frame in enumerate(buffer):
            processed_buffer.put(
                self._process_frame(
                    frame,
                    processor_index=processor_index,
                    source_sequence_index=source_sequence_index,
                    intermediate_store=intermediate_store,
                )
            )
        processed_buffer.close()
        return processed_buffer

    def _process_frame(
        self,
        frame: Frame,
        *,
        processor_index: int,
        source_sequence_index: int,
        intermediate_store: IntermediateFrameArtifactStore | None,
    ) -> Frame:
        processed_frame = self.single_frame_processor.process(frame)
        if intermediate_store is not None and intermediate_store.should_capture(source_sequence_index):
            self._capture_intermediate_artifacts(
                original_frame=frame,
                processed_frame=processed_frame,
                processor_index=processor_index,
                source_sequence_index=source_sequence_index,
                intermediate_store=intermediate_store,
            )
        return processed_frame

    def _capture_intermediate_artifacts(
        self,
        *,
        original_frame: Frame,
        processed_frame: Frame,
        processor_index: int,
        source_sequence_index: int,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> None:
        context = self._capture_context(
            processed_frame=processed_frame,
            processor_index=processor_index,
            source_sequence_index=source_sequence_index,
        )
        emitted = self._emit_from_single_frame_processor(original_frame, processed_frame, context)
        artifacts = emitted or (
            self._default_artifact(
                original_frame=original_frame,
                processed_frame=processed_frame,
                context=context,
                include_original=intermediate_store.config.include_original,
            ),
        )
        for artifact in artifacts:
            intermediate_store.add(artifact, source_sequence_index=source_sequence_index)

    def _capture_context(
        self,
        *,
        processed_frame: Frame,
        processor_index: int,
        source_sequence_index: int,
    ) -> IntermediateFrameCaptureContext:
        processor_name = type(self.single_frame_processor).__name__
        return IntermediateFrameCaptureContext(
            source_sequence_index=source_sequence_index,
            frame_index=processed_frame.index,
            single_frame_processor_index=processor_index,
            single_frame_processor_name=processor_name,
            stage_name=f"frame_processing[{processor_index}].{processor_name}",
            timestamp_seconds=processed_frame.timestamp_seconds,
            single_frame_processor_config=dict(getattr(self.single_frame_processor, "config", {}) or {}),
        )

    def _emit_from_single_frame_processor(
        self,
        original_frame: Frame,
        processed_frame: Frame,
        context: IntermediateFrameCaptureContext,
    ) -> tuple[IntermediateFrameArtifact, ...]:
        emitter = getattr(self.single_frame_processor, "emit_intermediate_artifacts", None)
        if not callable(emitter):
            return ()
        artifacts = tuple(emitter(original_frame, processed_frame, context))
        if any(not isinstance(artifact, IntermediateFrameArtifact) for artifact in artifacts):
            raise TypeError(
                f"{type(self.single_frame_processor).__name__}.emit_intermediate_artifacts must return IntermediateFrameArtifact instances."
            )
        return artifacts

    @staticmethod
    def _default_artifact(
        *,
        original_frame: Frame,
        processed_frame: Frame,
        context: IntermediateFrameCaptureContext,
        include_original: bool,
    ) -> IntermediateFrameArtifact:
        metadata: dict[str, Any] = {
            "single_frame_processor_name": context.single_frame_processor_name,
            "single_frame_processor_index": context.single_frame_processor_index,
            "source_sequence_index": context.source_sequence_index,
            "input_shape": tuple(int(dimension) for dimension in original_frame.image.shape),
            "output_shape": tuple(int(dimension) for dimension in processed_frame.image.shape),
        }
        if processed_frame.metadata:
            metadata["frame_metadata"] = dict(processed_frame.metadata)

        return IntermediateFrameArtifact(
            image=processed_frame.image,
            stage_name=context.stage_name,
            frame_index=context.frame_index,
            timestamp_seconds=context.timestamp_seconds,
            original_image=original_frame.image if include_original else None,
            stage_metadata=metadata,
            metadata=metadata,
            config=context.single_frame_processor_config,
        )
