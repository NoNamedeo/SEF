from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.BufferContracts import IBuffer
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import IStreamingFrameBufferProcessor
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from library.core.realtime.IRealtimeFrameSink import IRealtimeFrameSink
from library.core.realtime.NullRealtimeFrameSink import NullRealtimeFrameSink
from library.core.realtime.RealtimeFrame import RealtimeFrame


class RealtimeFrameTapProcessor(IStreamingFrameBufferProcessor):
    """
    Publish incoming frames to a realtime sink while forwarding them unchanged.

    This component is a generic stream tap: it does not know about Streamlit,
    YOLO, pose data, or any downstream analyzer. It exists to make live UIs
    responsive immediately, before slower analysis stages produce annotated data.
    """

    capabilities = StageCapabilities.streaming(
        stateful=False,
        preserves_order=True,
        realtime_safe=True,
    )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        sink: IRealtimeFrameSink | None = None,
    ) -> None:
        super().__init__(config)
        self._sink = sink or NullRealtimeFrameSink()
        self._publish_every_n_frames = max(1, int(self.config.get("publish_every_n_frames", 1)))

    def process(self, buffer: FrameBuffer) -> FrameBuffer:
        output = buffer.clone_empty()
        for index, frame in enumerate(buffer):
            self._publish_if_needed(frame, index)
            output.put(frame)
        output.close()
        return output

    def process_into(
        self,
        input_buffer: Iterable[Frame],
        output_buffer: IBuffer[Frame],
        *,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore | None,
    ) -> None:
        try:
            for index, frame in enumerate(input_buffer):
                if output_buffer.closed:
                    self._abort_upstream(input_buffer)
                    break
                self._publish_if_needed(frame, index)
                output_buffer.put(frame)
        finally:
            output_buffer.close()

    def _publish_if_needed(self, frame: Frame, sequence_index: int) -> None:
        if sequence_index % self._publish_every_n_frames != 0:
            return
        self._sink.publish(
            RealtimeFrame(
                image=frame.image,
                color_space="BGR",
                frame_index=frame.index,
                timestamp_seconds=frame.timestamp_seconds,
                metadata={
                    **dict(frame.metadata),
                    "preview_stage": "frame_tap",
                    "preview_priority": 10,
                },
            )
        )

    @staticmethod
    def _abort_upstream(input_buffer: Iterable[Frame]) -> None:
        abort = getattr(input_buffer, "abort", None)
        if callable(abort):
            abort()
