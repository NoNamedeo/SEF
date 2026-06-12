from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sef.builtin.analyzers.playback.TrackingPlaybackAnalyzer import TrackingPlaybackAnalyzer
from sef.builtin.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from sef.builtin.signal_extractors.OpenCVMultiObjectSignalExtractor import OpenCVMultiObjectSignalExtractor
from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from sef.core.pipeline.PipelineContext import PipelineContext


def moving_object_box(frame_index: int) -> tuple[int, int, int, int]:
    """Return the deterministic seed ROI used by the synthetic demo video."""
    return 24 + frame_index * 2, 34 + frame_index, 18, 18


def create_realistic_demo_video(path: str | Path, frame_count: int = 24) -> str:
    """Create a small deterministic multi-object tracking demo video."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_size = (160, 120)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        12.0,
        frame_size,
    )

    for frame_index in range(frame_count):
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        frame[:] = (18, 22, 28)

        for offset_x, offset_y in ((0, 0), (54, 8), (92, 34)):
            x, y, w, h = moving_object_box(frame_index)
            top_left = (min(x + offset_x, frame_size[0] - w - 1), min(y + offset_y, frame_size[1] - h - 1))
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(frame, top_left, bottom_right, (235, 235, 235), -1)
            cv2.line(frame, (top_left[0] + w // 2, top_left[1]), (top_left[0] + w // 2, bottom_right[1]), (80, 80, 80), 1)
            cv2.line(frame, (top_left[0], top_left[1] + h // 2), (bottom_right[0], top_left[1] + h // 2), (80, 80, 80), 1)

        writer.write(frame)

    writer.release()
    return str(output_path)


def build_realistic_sync_context(video_path: str | Path) -> PipelineContext:
    """Build the deterministic demo context used by tests and examples."""
    source_path = str(video_path)
    signal_extractor = OpenCVMultiObjectSignalExtractor(
        tracker_type="MIL",
        start_box=moving_object_box(0),
        max_objects=3,
        template_match_threshold=0.86,
        min_detection_distance=22,
        config={"source_path": source_path, "show": False},
    )

    return (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                path=source_path,
                buffer=FrameBuffer(32),
                config={"stride": 1, "max_frames": 24},
            )
        )
        .with_signal_extractor(signal_extractor)
        .add_analyzer(TrackingPlaybackAnalyzer())
        .build_context()
    )


__all__ = [
    "build_realistic_sync_context",
    "create_realistic_demo_video",
    "moving_object_box",
]
