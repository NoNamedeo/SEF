from __future__ import annotations

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.IData import IData
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineStageExecutor import PipelineStageExecutor


class SignalPipelineExecutor:
    """Runs the batch signal extraction, cleaning, and analysis tail."""

    def __init__(
        self,
        *,
        context: PipelineContext,
        stage_executor: PipelineStageExecutor,
    ) -> None:
        self._context = context
        self._stage_executor = stage_executor

    def run_batch(self, frames: FrameBuffer) -> list[IData]:
        """Extract a signal from materialized frames and run all analyzers."""
        signal = self._stage_executor.run("signal_extraction", lambda: self._context.signal_extractor.extract(frames))

        for cleaner_index, cleaner in enumerate(self._context.signal_cleaners):
            signal = self._stage_executor.run(
                f"signal_cleaning[{cleaner_index}]",
                lambda c=cleaner: c.clean(signal),
            )

        results: list[IData] = []
        for analyzer_index, analyzer in enumerate(self._context.analyzers):
            result = self._stage_executor.run(
                f"analysis[{analyzer_index}]",
                lambda a=analyzer: a.analyze(signal),
            )
            results.append(result)
        return results
