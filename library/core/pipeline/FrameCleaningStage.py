from __future__ import annotations

from collections.abc import Sequence

from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.IFrameCleaner import IFrameCleaner


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
    ) -> FrameBuffer:
        if not frame_cleaners:
            return buffer

        cleaned_buffer = buffer.clone_empty()
        for frame in buffer:
            cleaned_buffer.put(self._clean_frame(frame, frame_cleaners))
        cleaned_buffer.close()
        return cleaned_buffer

    @staticmethod
    def _clean_frame(frame: Frame, frame_cleaners: Sequence[IFrameCleaner]) -> Frame:
        cleaned_frame = frame
        for cleaner in frame_cleaners:
            cleaned_frame = cleaner.clean(cleaned_frame)
        return cleaned_frame
