from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.pipeline.PipelineBuilder import PipelineBuilder
from library.frame_cleaners.OpenCVGrayFrameCleaner import OpenCVGrayFrameCleaner
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.OpenCVMovingAverageCleaner import OpenCVMovingAverageCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor


class FakeTracker:
    def __init__(self):
        self._box = (0, 0, 0, 0)

    def init(self, _frame, box):
        self._box = box

    def update(self, _frame):
        x, y, w, h = self._box
        self._box = (x + 2, y + 3, w, h)
        return True, self._box


class PipelineCoreTests(unittest.TestCase):
    def test_gray_cleaner_preserves_frame_contract(self):
        color_image = np.zeros((8, 8, 3), dtype=np.uint8)
        frame = Frame(image=color_image, index=4, timestamp_seconds=0.2)

        cleaned = OpenCVGrayFrameCleaner().clean(frame)

        self.assertIsInstance(cleaned, Frame)
        self.assertEqual(cleaned.index, 4)
        self.assertEqual(cleaned.timestamp_seconds, 0.2)
        self.assertEqual(cleaned.frame.shape, (8, 8))

    def test_pipeline_runs_end_to_end_via_core_components(self):
        video_path = self._create_test_video()
        pipeline = (
            PipelineBuilder()
            .with_frame_extractor(
                OpenCVBufferedFrameExtractor(
                    path=video_path,
                    buffer=FrameBuffer(8),
                    config={"stride": 1, "max_frames": 4},
                )
            )
            .with_signal_extractor(
                OpenCVBufferedSignalExtractor(
                    tracker_type="MIL",
                    start_box=(5, 5, 10, 10),
                    tracker_factory=FakeTracker,
                )
            )
            .add_signal_cleaner(OpenCVMovingAverageCleaner(window_size=3))
            .add_analyzer(VerticalPositionAnalyzer(config={"use_timestamps": False}))
            .build()
        )

        results = pipeline.run()

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.title, "Vertical Position Over Time")
        self.assertEqual(result.x, [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(len(result.y), 4)
        self.assertTrue(all(isinstance(value, float) for value in result.y))

    def _create_test_video(self) -> str:
        temp_dir = Path(tempfile.mkdtemp())
        video_path = temp_dir / "synthetic.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (32, 32),
        )

        for frame_index in range(4):
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            cv2.rectangle(
                image,
                (4 + frame_index, 5 + frame_index),
                (14 + frame_index, 15 + frame_index),
                (255, 255, 255),
                -1,
            )
            writer.write(image)

        writer.release()
        return str(video_path)


if __name__ == "__main__":
    unittest.main()
