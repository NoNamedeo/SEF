from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from library.analyzers.ArucoMarkerDisplacementAnalyzer import ArucoMarkerDisplacementAnalyzer
from library.analyzers.ArucoMarkerRelativeMotionAnalyzer import ArucoMarkerRelativeMotionAnalyzer
from library.core.artifacts.ArucoDisplacementData import ArucoMarkerDisplacementData
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.Pipeline import Pipeline
from library.core.plugins.PluginRegistry import create_builtin_registry
from library.core.visualization.VisualArtifact import VideoArtifact
from library.signal_extractors.ArucoMarkerSignalExtractor import ArucoMarkerSignalExtractor
from library.visualizers.ArucoAnnotatedVideoVisualizer import ArucoAnnotatedVideoVisualizer
from library.visualizers.MatplotlibArucoMotionVisualizer import MatplotlibArucoMotionVisualizer


class ArucoPipelineTests(unittest.TestCase):
    def test_extractor_detects_markers_and_marks_missing_frames(self) -> None:
        buffer = self._build_frame_buffer(
            [
                {7: (40, 60)},
                {7: (52, 72)},
                {},
            ]
        )

        signal = ArucoMarkerSignalExtractor(marker_ids=[7]).extract(buffer)
        samples = list(signal)

        self.assertEqual(len(samples), 3)
        first_marker = samples[0].marker_by_id(7)
        self.assertIsNotNone(first_marker)
        self.assertTrue(first_marker.detected)
        self.assertEqual(len(first_marker.corners), 4)
        self.assertGreater(first_marker.quality_score, 0.0)

        missing_marker = samples[2].marker_by_id(7)
        self.assertIsNotNone(missing_marker)
        self.assertFalse(missing_marker.detected)
        self.assertIsNone(missing_marker.center)

    def test_extractor_detects_full_frame_generated_marker_with_padding_fallback(self) -> None:
        marker = self._generate_marker_image(3, 500)
        frame = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        buffer = FrameBuffer(1)
        buffer.put(Frame(image=frame, index=0, timestamp_seconds=0.0))
        buffer.close()

        signal = ArucoMarkerSignalExtractor(marker_ids=[3]).extract(buffer)
        samples = list(signal)

        observation = samples[0].marker_by_id(3)
        self.assertIsNotNone(observation)
        self.assertTrue(observation.detected)
        self.assertIsNotNone(observation.center)

    def test_displacement_analyzer_computes_expected_motion(self) -> None:
        signal = ArucoMarkerSignalExtractor(marker_ids=[7]).extract(
            self._build_frame_buffer(
                [
                    {7: (40, 60)},
                    {7: (50, 70)},
                    {},
                ]
            )
        )

        data = ArucoMarkerDisplacementAnalyzer(marker_ids=[7]).analyze(signal)
        series = data.series[0]

        self.assertEqual(series.marker_id, 7)
        self.assertAlmostEqual(series.displacement_x[0], 0.0, places=3)
        self.assertAlmostEqual(series.displacement_y[0], 0.0, places=3)
        self.assertAlmostEqual(series.displacement_x[1], 10.0, delta=2.0)
        self.assertAlmostEqual(series.displacement_y[1], 10.0, delta=2.0)
        self.assertAlmostEqual(series.displacement_magnitude[1], math.hypot(10.0, 10.0), delta=3.0)
        self.assertTrue(math.isnan(series.displacement_x[2]))
        self.assertEqual(series.stats["detected_samples"], 2.0)

    def test_relative_motion_analyzer_computes_distance_delta(self) -> None:
        signal = ArucoMarkerSignalExtractor(marker_ids=[7, 8]).extract(
            self._build_frame_buffer(
                [
                    {7: (30, 70), 8: (140, 70)},
                    {7: (42, 70), 8: (140, 70)},
                ]
            )
        )

        data = ArucoMarkerRelativeMotionAnalyzer(marker_pairs=[(7, 8)]).analyze(signal)
        series = data.series[0]

        self.assertEqual(series.marker_pair, (7, 8))
        self.assertAlmostEqual(series.distance_deltas[0], 0.0, delta=1.5)
        self.assertAlmostEqual(series.distance_deltas[1], -12.0, delta=3.0)

    def test_builtin_registry_exposes_aruco_components(self) -> None:
        registry = create_builtin_registry()

        extractor = registry.create("signal_extractor", "aruco_marker", marker_ids=[7])
        displacement_analyzer = registry.create("analyzer", "aruco_displacement")
        relative_analyzer = registry.create("analyzer", "aruco_relative_motion")
        video_visualizer = registry.create("visualizer", "aruco_annotated_video")
        plot_visualizer = registry.create("visualizer", "aruco_motion_plot")

        self.assertIsInstance(extractor, ArucoMarkerSignalExtractor)
        self.assertIsInstance(displacement_analyzer, ArucoMarkerDisplacementAnalyzer)
        self.assertIsInstance(relative_analyzer, ArucoMarkerRelativeMotionAnalyzer)
        self.assertIsInstance(video_visualizer, ArucoAnnotatedVideoVisualizer)
        self.assertIsInstance(plot_visualizer, MatplotlibArucoMotionVisualizer)

    def test_config_builder_runs_aruco_pipeline_with_video_artifact(self) -> None:
        video_path = self._create_test_video(
            [
                {7: (40, 60)},
                {7: (48, 66)},
                {7: (55, 70)},
            ]
        )
        registry = create_builtin_registry()
        config = {
            "pipeline": {
                "frame_extractor": {
                    "name": "opencv_buffered",
                    "params": {
                        "path": video_path,
                        "config": {"stride": 1, "max_frames": 3},
                    },
                },
                "signal_extractor": {
                    "name": "aruco_marker",
                    "params": {"marker_ids": [7]},
                },
                "analyzers": [
                    {"name": "aruco_displacement"},
                ],
                "visualizers": [
                    {"name": "aruco_annotated_video", "result_indices": [0]},
                ],
            }
        }

        context = ConfigPipelineBuilder(registry).build_context(config)
        outputs = Pipeline(context).run()

        self.assertEqual(len(outputs.results), 1)
        self.assertIsInstance(outputs.results[0], ArucoMarkerDisplacementData)
        self.assertEqual(len(outputs.artifacts), 1)
        self.assertIsInstance(outputs.artifacts[0], VideoArtifact)
        self.assertGreater(len(outputs.artifacts[0].data), 0)

    def _build_frame_buffer(
        self,
        marker_frames: list[dict[int, tuple[int, int]]],
    ) -> FrameBuffer:
        buffer = FrameBuffer(len(marker_frames))
        for index, marker_positions in enumerate(marker_frames):
            buffer.put(
                Frame(
                    image=self._build_frame(marker_positions),
                    index=index,
                    timestamp_seconds=index * 0.1,
                    metadata={"source_path": "/tmp/aruco_test.mp4", "source_fps": 10.0},
                )
            )
        buffer.close()
        return buffer

    def _create_test_video(
        self,
        marker_frames: list[dict[int, tuple[int, int]]],
    ) -> str:
        temp_dir = Path(tempfile.mkdtemp())
        video_path = temp_dir / "aruco_demo.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (240, 240),
        )
        for marker_positions in marker_frames:
            writer.write(self._build_frame(marker_positions))
        writer.release()
        return str(video_path)

    def _build_frame(self, marker_positions: dict[int, tuple[int, int]]) -> np.ndarray:
        canvas = np.full((240, 240), 255, dtype=np.uint8)
        marker_side = 72
        for marker_id, (x, y) in marker_positions.items():
            marker = self._generate_marker_image(marker_id, marker_side)
            canvas[y : y + marker_side, x : x + marker_side] = marker
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _generate_marker_image(marker_id: int, side: int) -> np.ndarray:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        if hasattr(cv2.aruco, "generateImageMarker"):
            return cv2.aruco.generateImageMarker(dictionary, marker_id, side)

        marker = np.zeros((side, side), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, side, marker, 1)
        return marker


if __name__ == "__main__":
    unittest.main()
