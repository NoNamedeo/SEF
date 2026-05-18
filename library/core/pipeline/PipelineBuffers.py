from __future__ import annotations

from typing import Any

from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.FrameBuffer import FrameBuffer


class PipelineBuffers:
    """
    Utility operations for bounded buffers used during pipeline execution.

    Buffer operations are kept here because they are cross-cutting primitives:
    frame execution, materialization and streaming failure handling all need
    them, but no runtime collaborator should own generic buffer mechanics.
    """

    @staticmethod
    def copy_frame_buffer(source_buffer: FrameBuffer) -> FrameBuffer:
        """
        Materialize a frame stream into a replayable buffer.

        A new buffer is returned so downstream batch-only components can iterate
        independently from the producer stream that fed the materialization.
        The returned buffer is closed and sized to hold all copied frames plus a
        sentinel slot when needed by the underlying bounded-buffer contract.
        """
        frames = list(source_buffer)
        output = FrameBuffer(buffer_size=max(len(frames) + 1, source_buffer.capacity))
        for frame in frames:
            output.put(frame)
        output.close()
        return output

    @staticmethod
    def abort_all(
        frame_buffers: list[FrameBuffer],
        signal_buffers: list[Any],
        data_buffers: list[DataBuffer],
    ) -> None:
        """
        Unblock producers and consumers after a concurrent stage fails.

        The method intentionally uses duck typing for ``abort`` because frame,
        signal and data buffers share the behavior without requiring a common
        inheritance hierarchy.
        """
        for buffer in [*frame_buffers, *signal_buffers, *data_buffers]:
            abort = getattr(buffer, "abort", None)
            if callable(abort):
                abort()
