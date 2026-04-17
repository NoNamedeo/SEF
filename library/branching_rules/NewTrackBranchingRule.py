from __future__ import annotations

import logging
from pathlib import Path

from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.events.Event import Event
from library.core.interfaces.pipeline.IBranchingRule import IBranchingRule
from library.core.pipeline.PipelineContext import PipelineContext
from library.frame_cleaners.SmoothingFrameCleaner import SmoothingFrameCleaner
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.MovingAverageCleaner import MovingAverageCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor

log = logging.getLogger(__name__)


class NewTrackBranchingRule(IBranchingRule):
    """Branch once when the primary multi-object tracker creates its seed track."""

    def matches(self, event: Event) -> bool:
        if event.event_type != "track_created":
            return False
        if str(event.payload.get("kind", "")) != "seed":
            return False
        pipeline_id = str(event.payload.get("pipeline_id", ""))
        return not pipeline_id.startswith("secondary-")

    def build_context(self, event: Event) -> PipelineContext:
        source_path = event.require("source_path")
        box = event.require("box")
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise ValueError("track_created payload field 'box' must be a 4-item box.")

        start_box = tuple(int(round(float(value))) for value in box)
        if source_path and not Path(str(source_path)).exists():
            log.warning("Branching source video not found on disk: %s", source_path)

        frame_extractor = OpenCVBufferedFrameExtractor(
            path=str(source_path),
            config={"resize": None, "stride": 2, "max_frames": 180},
        )

        signal_extractor = OpenCVBufferedSignalExtractor(
            tracker_type="CSRT",
            start_box=start_box,
            config={"show": False},
        )

        return PipelineContext(
            frame_extractor=frame_extractor,
            signal_extractor=signal_extractor,
            frame_cleaners=(SmoothingFrameCleaner(),),
            signal_cleaners=(MovingAverageCleaner(window_size=5),),
            analyzers=(VerticalPositionAnalyzer(),),
        )
