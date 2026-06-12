from __future__ import annotations

from sef.core.pipeline.PipelineComponentCapabilities import PipelineComponentCapabilities
from sef.core.pipeline.PipelineContext import PipelineContext


class PipelineExecutionLookahead:
    """
    Answers downstream streamability questions for planning and execution.

    The class is intentionally read-only. It keeps stage-order knowledge in one
    place so the execution planner and segmented runtime cannot drift when they
    decide whether opening a streaming segment is useful.
    """

    def __init__(self, context: PipelineContext) -> None:
        self._context = context

    def frame_successor_streamable(self, *, processor_index: int) -> bool:
        """
        Return True when the next frame-side stage can consume streaming frames.

        ``processor_index`` points to the next frame processor that would run.
        When there are no more processors, the lookahead continues into frame
        exporters and then into signal extraction.
        """
        if processor_index < len(self._context.frame_processors):
            return PipelineComponentCapabilities.can_stream_frame_processor(
                self._context.frame_processors[processor_index]
            )
        return self.frame_export_successor_streamable(exporter_index=0)

    def frame_export_successor_streamable(self, *, exporter_index: int) -> bool:
        """
        Return True when the next frame-export or signal stage can stream.

        ``exporter_index`` points to the next exporter that would run. Once all
        exporters are consumed, the frame-side lookahead crosses into signal
        extraction because a streaming extractor can consume a frame stream.
        """
        if exporter_index < len(self._context.frame_exporters):
            return PipelineComponentCapabilities.can_stream_frame_exporter(
                self._context.frame_exporters[exporter_index]
            )
        return PipelineComponentCapabilities.can_stream_signal_extractor(
            self._context.signal_extractor
        )

    def signal_successor_streamable(self, *, cleaner_index: int) -> bool:
        """
        Return True when the next signal-side stage can consume streaming data.

        ``cleaner_index`` points to the next cleaner that would run. Once all
        cleaners are consumed, streamability depends on whether at least one
        analyzer can consume progressive signal samples.
        """
        if cleaner_index < len(self._context.signal_cleaners):
            return PipelineComponentCapabilities.can_stream_signal_cleaner(
                self._context.signal_cleaners[cleaner_index]
            )
        return any(
            PipelineComponentCapabilities.can_stream_analyzer(analyzer)
            for analyzer in self._context.analyzers
        )
