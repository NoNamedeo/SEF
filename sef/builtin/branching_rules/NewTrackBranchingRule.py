from __future__ import annotations

import logging
from pathlib import Path

from sef.core.events.Event import Event
from sef.core.interfaces.pipeline.IBranchingRule import IBranchingRule

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

    def build_config(self, event: Event) -> dict:
        source_path = event.require("source_path")
        box = event.require("box")
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise ValueError("track_created payload field 'box' must be a 4-item box.")

        start_box = tuple(int(round(float(value))) for value in box)
        if source_path and not Path(str(source_path)).exists():
            log.warning("Branching source video not found on disk: %s", source_path)

        return {
            "pipeline": {
                "frame_extractor": {
                    "name": "opencv_buffered",
                    "params": {
                        "path": str(source_path),
                        "config": {"resize": None, "stride": 2, "max_frames": 180},
                    },
                },
                "frame_processors": [
                    {"name": "smoothing", "processor_type": "single_frame"},
                ],
                "signal_extractor": {
                    "name": "opencv_tracker",
                    "params": {
                        "tracker_type": "CSRT",
                        "start_box": start_box,
                        "config": {"show": False},
                    },
                },
                "signal_cleaners": [
                    {"name": "moving_average", "params": {"window_size": 5}},
                ],
                "analyzers": [{"name": "vertical_position"}],
            },
        }
