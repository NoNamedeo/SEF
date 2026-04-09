from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.core.pipeline.Pipeline import Pipeline


class PipelineValidator:
    """Default validator for the synchronous SEF pipeline."""

    def validate(self, pipeline: "Pipeline") -> None:
        if pipeline.frame_extractor is None:
            raise ValueError("Frame extractor not valid")
        if pipeline.signal_extractor is None:
            raise ValueError("Signal extractor not valid")
        if not pipeline.analyzers:
            raise ValueError("Analyzer list not valid")
