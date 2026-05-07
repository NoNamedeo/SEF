from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.MaskArtifacts import IntermediateFrameArtifact
from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.pipeline.IntermediateFrameCapture import (
    IntermediateFrameArtifactStore,
    IntermediateFrameCaptureContext,
)


class FrameCleaningStage:
    """
    Apply frame cleaners inside the pipeline boundary.

    The extractor produces raw frames only. This stage owns the ordered
    application of IFrameCleaner implementations and returns a new buffer
    with the cleaned frames.
    """

    def apply(
        self,
        buffer: FrameBuffer,
        frame_cleaners: Sequence[IFrameCleaner],
        intermediate_store: IntermediateFrameArtifactStore | None = None,
    ) -> FrameBuffer:
        if not frame_cleaners:
            return buffer

        cleaned_buffer = buffer.clone_empty()
        for source_sequence_index, frame in enumerate(buffer):
            cleaned_buffer.put(
                self._clean_frame(
                    frame,
                    frame_cleaners,
                    intermediate_store=intermediate_store,
                    source_sequence_index=source_sequence_index,
                )
            )
        cleaned_buffer.close()
        return cleaned_buffer

    @staticmethod
    def _clean_frame(
        frame: Frame,
        frame_cleaners: Sequence[IFrameCleaner],
        *,
        intermediate_store: IntermediateFrameArtifactStore | None = None,
        source_sequence_index: int = 0,
    ) -> Frame:
        cleaned_frame = frame
        for cleaner_index, cleaner in enumerate(frame_cleaners):
            original_frame = cleaned_frame
            cleaned_frame = cleaner.clean(cleaned_frame)
            if intermediate_store is not None and intermediate_store.should_capture(source_sequence_index):
                FrameCleaningStage._capture_intermediate_artifacts(
                    cleaner=cleaner,
                    original_frame=original_frame,
                    cleaned_frame=cleaned_frame,
                    cleaner_index=cleaner_index,
                    source_sequence_index=source_sequence_index,
                    intermediate_store=intermediate_store,
                )
        return cleaned_frame

    @staticmethod
    def _capture_intermediate_artifacts(
        *,
        cleaner: IFrameCleaner,
        original_frame: Frame,
        cleaned_frame: Frame,
        cleaner_index: int,
        source_sequence_index: int,
        intermediate_store: IntermediateFrameArtifactStore,
    ) -> None:
        context = FrameCleaningStage._capture_context(
            cleaner=cleaner,
            cleaned_frame=cleaned_frame,
            cleaner_index=cleaner_index,
            source_sequence_index=source_sequence_index,
        )
        emitted = FrameCleaningStage._emit_from_cleaner(
            cleaner,
            original_frame,
            cleaned_frame,
            context,
        )
        artifacts = emitted or (
            FrameCleaningStage._default_artifact(
                original_frame=original_frame,
                cleaned_frame=cleaned_frame,
                context=context,
                include_original=intermediate_store.config.include_original,
            ),
        )
        for artifact in artifacts:
            intermediate_store.add(
                artifact,
                source_sequence_index=source_sequence_index,
            )

    @staticmethod
    def _capture_context(
        *,
        cleaner: IFrameCleaner,
        cleaned_frame: Frame,
        cleaner_index: int,
        source_sequence_index: int,
    ) -> IntermediateFrameCaptureContext:
        cleaner_name = type(cleaner).__name__
        return IntermediateFrameCaptureContext(
            source_sequence_index=source_sequence_index,
            frame_index=cleaned_frame.index,
            cleaner_index=cleaner_index,
            cleaner_name=cleaner_name,
            stage_name=f"frame_cleaning[{cleaner_index}].{cleaner_name}",
            timestamp_seconds=cleaned_frame.timestamp_seconds,
            cleaner_config=dict(getattr(cleaner, "config", {}) or {}),
        )

    @staticmethod
    def _emit_from_cleaner(
        cleaner: IFrameCleaner,
        original_frame: Frame,
        cleaned_frame: Frame,
        context: IntermediateFrameCaptureContext,
    ) -> tuple[IntermediateFrameArtifact, ...]:
        emitter = getattr(cleaner, "emit_intermediate_artifacts", None)
        if not callable(emitter):
            return ()
        artifacts = tuple(emitter(original_frame, cleaned_frame, context))
        if any(not isinstance(artifact, IntermediateFrameArtifact) for artifact in artifacts):
            raise TypeError(
                f"{type(cleaner).__name__}.emit_intermediate_artifacts must return IntermediateFrameArtifact instances."
            )
        return artifacts

    @staticmethod
    def _default_artifact(
        *,
        original_frame: Frame,
        cleaned_frame: Frame,
        context: IntermediateFrameCaptureContext,
        include_original: bool,
    ) -> IntermediateFrameArtifact:
        metadata: dict[str, Any] = {
            "cleaner_name": context.cleaner_name,
            "cleaner_index": context.cleaner_index,
            "source_sequence_index": context.source_sequence_index,
            "input_shape": tuple(int(dimension) for dimension in original_frame.image.shape),
            "output_shape": tuple(int(dimension) for dimension in cleaned_frame.image.shape),
        }
        if cleaned_frame.metadata:
            metadata["frame_metadata"] = dict(cleaned_frame.metadata)

        return IntermediateFrameArtifact(
            image=cleaned_frame.image,
            stage_name=context.stage_name,
            frame_index=context.frame_index,
            timestamp_seconds=context.timestamp_seconds,
            original_image=original_frame.image if include_original else None,
            stage_metadata=metadata,
            metadata=metadata,
            config=context.cleaner_config,
        )
