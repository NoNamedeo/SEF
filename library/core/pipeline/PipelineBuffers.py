from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.BufferContracts import IAbortableBuffer, IFrameBuffer


class PipelineBuffers:
    """
    Utility operations for bounded buffers used during pipeline execution.

    Buffer operations are kept here because they are cross-cutting primitives:
    frame execution, materialization and streaming failure handling all need
    them, but no runtime collaborator should own generic buffer mechanics.
    """

    @staticmethod
    def copy_frame_buffer(source_buffer: IFrameBuffer) -> FrameBuffer:
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
        *buffer_groups: Iterable[IAbortableBuffer[Any]],
    ) -> None:
        """
        Unblock producers and consumers after a concurrent stage fails.

        Failure handling depends only on the abortable-buffer contract, so new
        buffer implementations can join the runtime without changing this helper.
        """
        for buffers in buffer_groups:
            for buffer in buffers:
                buffer.abort()
