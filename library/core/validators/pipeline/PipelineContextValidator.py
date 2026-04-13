from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.core.pipeline.Pipeline import PipelineContext


class PipelineContextValidator:
    @staticmethod
    def validate(context: PipelineContext) -> None:
        missing = []

        if context.frame_extractor is None:
            missing.append("frame_extractor")
        if context.signal_extractor is None:
            missing.append("signal_extractor")
        if not context.analyzers:
            missing.append("analyzers")

        if missing:
            raise ValueError(f"Invalid PipelineContext: {missing}")
