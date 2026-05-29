from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from library.analyzers.playback.TrackingPlaybackAnalyzer import TrackingPlaybackAnalyzer
from library.analyzers.single_tracker.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.artifacts.Frame import Frame
from library.core.artifacts.buffer.FrameBuffer import FrameBuffer
from library.core.artifacts.signal_sample.MultiObjectSignalSample import MultiObjectSignalSample, MultiObjectTrack
from library.core.artifacts.Signal import Signal
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.Pipeline import Pipeline
from library.core.visualization.VisualArtifact import (
    VIDEO_ARTIFACT_TYPES,
    DeferredVideoArtifact,
    VideoArtifact,
    VideoFileArtifact,
)
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.frame_processors.OpenCV.OpenCVGrayFrameProcessor import OpenCVGrayFrameProcessor
from library.Main import build_realistic_sync_context, create_realistic_demo_video, moving_object_box
from library.signal_cleaners.single_tracker.MovingAverageCleaner import MovingAverageCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.signal_extractors.OpenCVMultiObjectSignalExtractor import OpenCVMultiObjectSignalExtractor
from library.visualizers.TrackingVideoVisualizer import TrackingVideoVisualizer


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
    def test_gray_processor_preserves_frame_contract(self):
        color_image = np.zeros((8, 8, 3), dtype=np.uint8)
        frame = Frame(image=color_image, index=4, timestamp_seconds=0.2)

        processed = OpenCVGrayFrameProcessor().process(frame)

        self.assertIsInstance(processed, Frame)
        self.assertEqual(processed.index, 4)
        self.assertEqual(processed.timestamp_seconds, 0.2)
        self.assertEqual(processed.frame.shape, (8, 8))

    def test_pipeline_runs_end_to_end_via_core_components(self):
        video_path = self._create_test_video()
        context = (
            FluentPipelineBuilder()
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
            .add_signal_cleaner(MovingAverageCleaner(window_size=3))
            .add_analyzer(VerticalPositionAnalyzer(config={"use_timestamps": False}))
            .build_context()
        )
        pipeline = Pipeline(context)

        outputs = pipeline.run()

        self.assertEqual(len(outputs.results), 1)
        result = outputs.results[0]
        self.assertEqual(result.title, "Vertical Position Over Time")
        self.assertEqual(result.x, [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(len(result.y), 4)
        self.assertTrue(all(isinstance(value, float) for value in result.y))

    def test_multiobject_extractor_accepts_similarity_threshold_alias(self):
        buffer = FrameBuffer(2)
        for frame_index in range(2):
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            cv2.rectangle(image, (2, 2), (6, 6), (255, 255, 255), -1)
            buffer.put(Frame(image=image, index=frame_index, timestamp_seconds=frame_index * 0.1))
        buffer.close()

        extractor = OpenCVMultiObjectSignalExtractor(
            tracker_type="MIL",
            start_box=(2, 2, 4, 4),
            max_objects=1,
            similarity_threshold=0.7,
            tracker_factory=FakeTracker,
            config={"show": False},
        )

        signal = extractor.extract(buffer)
        samples = list(signal)

        self.assertEqual(len(samples), 2)
        self.assertEqual(extractor.template_match_threshold, 0.7)
        self.assertTrue(samples[0].tracks)
        self.assertEqual(samples[0].tracks[0].box, (4, 5, 4, 4))

    def test_realistic_multiobject_demo_uses_deterministic_seed_box(self):
        temp_dir = Path(tempfile.mkdtemp())
        video_path = create_realistic_demo_video(temp_dir / "demo.avi")

        context = build_realistic_sync_context(video_path)

        self.assertEqual(context.signal_extractor.start_box, moving_object_box(0))
        self.assertEqual(context.signal_extractor.template_match_threshold, 0.86)

    def test_tracking_playback_analyzer_maps_multiobject_samples(self):
        signal = Signal(
            [
                MultiObjectSignalSample(
                    frame_index=3,
                    tracks=[
                        MultiObjectTrack(track_id=4, box=(10, 12, 20, 14), centroid=(20.0, 19.0)),
                    ],
                    metadata={
                        "source_path": "/tmp/demo.mp4",
                        "resize": (320, 180),
                        "source_fps": 25.0,
                    },
                )
            ]
        )

        result = TrackingPlaybackAnalyzer().analyze(signal)

        self.assertEqual(result.source_path, "/tmp/demo.mp4")
        self.assertEqual(result.resize, (320, 180))
        self.assertEqual(result.fps, 25.0)
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].tracks[0].track_id, 4)
        self.assertEqual(result.frames[0].tracks[0].box, (10, 12, 20, 14))

    def test_tracking_video_visualizer_emits_video_artifact(self):
        video_path = self._create_test_video()
        playback_data = TrackingPlaybackAnalyzer().analyze(
            Signal(
                [
                    MultiObjectSignalSample(
                        frame_index=0,
                        tracks=[
                            MultiObjectTrack(track_id=0, box=(4, 5, 10, 10), centroid=(9.0, 10.0)),
                        ],
                        metadata={"source_path": video_path, "source_fps": 10.0},
                    ),
                    MultiObjectSignalSample(
                        frame_index=1,
                        tracks=[
                            MultiObjectTrack(track_id=0, box=(5, 6, 10, 10), centroid=(10.0, 11.0)),
                        ],
                        metadata={"source_path": video_path, "source_fps": 10.0},
                    ),
                ]
            )
        )

        artifacts = TrackingVideoVisualizer().render(playback_data)

        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertIsInstance(artifact, DeferredVideoArtifact)
        self.assertEqual(artifact.mime_type, "video/mp4")
        self.assertTrue(self._materialized_video_size(artifact) > 0)

    def test_tracking_video_visualizer_can_render_eager_downscaled_file(self):
        video_path = self._create_test_video()
        playback_data = TrackingPlaybackAnalyzer().analyze(
            Signal(
                [
                    MultiObjectSignalSample(
                        frame_index=0,
                        tracks=[
                            MultiObjectTrack(track_id=0, box=(4, 5, 10, 10), centroid=(9.0, 10.0)),
                        ],
                        metadata={"source_path": video_path, "source_fps": 10.0},
                    ),
                    MultiObjectSignalSample(
                        frame_index=1,
                        tracks=[
                            MultiObjectTrack(track_id=0, box=(5, 6, 10, 10), centroid=(10.0, 11.0)),
                        ],
                        metadata={"source_path": video_path, "source_fps": 10.0},
                    ),
                    MultiObjectSignalSample(
                        frame_index=2,
                        tracks=[
                            MultiObjectTrack(track_id=0, box=(6, 7, 10, 10), centroid=(11.0, 12.0)),
                        ],
                        metadata={"source_path": video_path, "source_fps": 10.0},
                    ),
                ]
            )
        )

        artifact = TrackingVideoVisualizer(
            config={
                "lazy": False,
                "frame_sample_interval": 2,
                "output_size": (16, 16),
                "codec": "mp4v",
            }
        ).render(playback_data)[0]

        self.assertIsInstance(artifact, VideoFileArtifact)
        self.assertEqual(artifact.metadata["rendered_frame_count"], 2)
        capture = cv2.VideoCapture(str(artifact.path))
        try:
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 16)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 16)
        finally:
            capture.release()

    def test_pipeline_can_emit_tracking_video_artifact(self):
        video_path = self._create_test_video()
        context = (
            FluentPipelineBuilder()
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
                    config={"source_path": video_path},
                )
            )
            .add_analyzer(TrackingPlaybackAnalyzer())
            .add_visualizer_for_results(TrackingVideoVisualizer(), [0])
            .build_context()
        )

        outputs = Pipeline(context).run()

        self.assertEqual(len(outputs.results), 1)
        self.assertEqual(len(outputs.final_artifacts), 1)
        self.assertIsInstance(outputs.final_artifacts[0], VIDEO_ARTIFACT_TYPES)
        self.assertTrue(self._materialized_video_size(outputs.final_artifacts[0]) > 0)

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

    @staticmethod
    def _materialized_video_size(artifact) -> int:
        if isinstance(artifact, VideoArtifact):
            return len(artifact.data)
        if isinstance(artifact, VideoFileArtifact):
            return artifact.path.stat().st_size
        if isinstance(artifact, DeferredVideoArtifact):
            return artifact.materialize().stat().st_size
        raise AssertionError(f"Unsupported video artifact: {type(artifact).__name__}")


if __name__ == "__main__":
    unittest.main()
