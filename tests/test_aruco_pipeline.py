from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sef.builtin.analyzers.ArUco.ArucoMarkerDisplacementAnalyzer import ArucoMarkerDisplacementAnalyzer
from sef.builtin.analyzers.ArUco.ArucoMarkerRelativeMotionAnalyzer import ArucoMarkerRelativeMotionAnalyzer
from sef.builtin.signal_cleaners.ArUco.ArucoTemporalStabilizerCleaner import ArucoTemporalStabilizerCleaner
from sef.builtin.signal_extractors.ArucoMarkerSignalExtractor import ArucoMarkerSignalExtractor
from sef.builtin.visualizers.ArUco.ArucoAnnotatedVideoVisualizer import ArucoAnnotatedVideoVisualizer
from sef.builtin.visualizers.Matplotlib.MatplotlibArucoMotionVisualizer import MatplotlibArucoMotionVisualizer
from sef.core.artifacts.buffer.DataBuffer import DataBuffer
from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.artifacts.data.ArucoDisplacementData import (
    ArucoMarkerDisplacementData,
    ArucoMarkerDisplacementFrameData,
)
from sef.core.artifacts.Frame import Frame
from sef.core.artifacts.Signal import Signal
from sef.core.artifacts.signal_sample.ArucoMarkerSignalSample import ArucoMarkerObservation, ArucoMarkerSignalSample
from sef.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from sef.core.pipeline.Pipeline import Pipeline
from sef.builtin.registry import create_builtin_registry
from sef.core.visualization.VisualArtifact import (
    VIDEO_ARTIFACT_TYPES,
    DeferredVideoArtifact,
    VideoArtifact,
    VideoFileArtifact,
)


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

    def test_extractor_detects_configured_4x4_dictionary_marker(self) -> None:
        marker = self._generate_marker_image(12, 96, dictionary_id=cv2.aruco.DICT_4X4_50)
        canvas = np.full((180, 180), 255, dtype=np.uint8)
        canvas[42:138, 42:138] = marker
        buffer = FrameBuffer(1)
        buffer.put(Frame(image=cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), index=0, timestamp_seconds=0.0))
        buffer.close()

        signal = ArucoMarkerSignalExtractor(
            marker_ids=[12],
            config={"aruco_dictionary": "DICT_4X4_50"},
        ).extract(buffer)

        observation = list(signal)[0].marker_by_id(12)
        self.assertIsNotNone(observation)
        self.assertTrue(observation.detected)
        self.assertEqual(list(signal)[0].metadata["aruco_dictionary"], "DICT_4X4_50")

    def test_extractor_accepts_short_dictionary_alias(self) -> None:
        extractor = ArucoMarkerSignalExtractor(config={"aruco_dictionary": "4x4_50"})

        self.assertEqual(extractor._dictionary_name, "DICT_4X4_50")

    def test_extractor_rejects_unknown_dictionary(self) -> None:
        with self.assertRaises(ValueError):
            ArucoMarkerSignalExtractor(config={"aruco_dictionary": "DICT_UNKNOWN"})

    def test_extractor_supports_manual_subpixel_refinement_and_quality_metadata(self) -> None:
        signal = ArucoMarkerSignalExtractor(
            marker_ids=[7],
            config={
                "corner_refinement_enabled": True,
                "corner_refinement_method": "manual_subpix",
                "corner_refinement_win_size": 5,
                "corner_refinement_max_iterations": 30,
                "corner_refinement_min_accuracy": 0.01,
            },
        ).extract(
            self._build_frame_buffer(
                [
                    {7: (40, 60)},
                ]
            )
        )

        observation = list(signal)[0].marker_by_id(7)

        self.assertIsNotNone(observation)
        self.assertTrue(observation.detected)
        self.assertEqual(observation.metadata["quality_model"], "aruco_area_border_shape_v1")
        self.assertEqual(observation.metadata["refinement_method"], "manual_subpix")
        self.assertTrue(observation.metadata["refinement_applied"])
        self.assertIn("quality_components", observation.metadata)
        self.assertIn("shape_score", observation.metadata["quality_components"])

    def test_extractor_rejects_pose_estimation_until_dedicated_upgrade_is_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            ArucoMarkerSignalExtractor(
                marker_ids=[7],
                config={
                    "estimate_pose": True,
                    "marker_length": 0.05,
                    "camera_matrix": [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]],
                    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                },
            )

    def test_temporal_stabilizer_reduces_center_jitter_for_quality_observations(self) -> None:
        raw_signal = Signal(
            [
                self._aruco_sample(0, 100.0, 100.0, quality=0.9),
                self._aruco_sample(1, 101.0, 99.0, quality=0.9),
                self._aruco_sample(2, 99.0, 101.0, quality=0.9),
                self._aruco_sample(3, 100.5, 99.5, quality=0.9),
            ]
        )

        cleaned_signal = ArucoTemporalStabilizerCleaner().clean(raw_signal)

        raw_centers = [sample.marker_by_id(7).center for sample in raw_signal]
        cleaned_centers = [sample.marker_by_id(7).center for sample in cleaned_signal]

        self.assertGreater(self._center_span(raw_centers), self._center_span(cleaned_centers))
        self.assertTrue(cleaned_signal.signal[1].marker_by_id(7).metadata["temporal_stabilizer"]["applied"])

    def test_temporal_stabilizer_is_registered_in_builtin_registry(self) -> None:
        registry = create_builtin_registry()
        cleaner = registry.create("signal_cleaner", "aruco_temporal_stabilizer")
        self.assertIsInstance(cleaner, ArucoTemporalStabilizerCleaner)

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

    def test_displacement_analyzer_streams_progressive_frame_data(self) -> None:
        raw_signal = Signal(
            [
                self._aruco_sample(0, 100.0, 100.0, quality=0.9),
                self._aruco_sample(1, 103.0, 104.0, quality=0.9),
                self._aruco_sample(2, 108.0, 109.0, quality=0.9),
            ]
        )
        buffer = DataBuffer(buffer_size=8)

        data = ArucoMarkerDisplacementAnalyzer(marker_ids=[7]).analyze_into(raw_signal, buffer)
        progressive_items = list(buffer.subscribe(0))

        self.assertIsInstance(data, ArucoMarkerDisplacementData)
        self.assertEqual(len(progressive_items), 3)
        self.assertTrue(all(isinstance(item, ArucoMarkerDisplacementFrameData) for item in progressive_items))
        self.assertAlmostEqual(progressive_items[1].displacements[7].displacement_x, 3.0)
        self.assertAlmostEqual(progressive_items[1].displacements[7].displacement_y, 4.0)

    def test_aruco_execution_plan_is_streamable_with_builtin_outputs(self) -> None:
        registry = create_builtin_registry()
        context = ConfigPipelineBuilder(registry).build_context(
            {
                "pipeline": {
                    "runtime": {
                        "frame_buffer_size": 4,
                        "signal_buffer_size": 4,
                        "data_buffer_size": 4,
                        "latency_policy": {"name": "blocking", "params": {}},
                    },
                    "frame_extractor": {
                        "name": "opencv_buffered",
                        "params": {
                            "path": "/tmp/unused_aruco_plan.avi",
                            "config": {"resize": [240, 240], "stride": 1, "max_frames": 8},
                        },
                    },
                    "signal_extractor": {
                        "name": "aruco_marker",
                        "params": {"marker_ids": [7]},
                    },
                    "signal_cleaners": [
                        {"name": "aruco_temporal_stabilizer"},
                    ],
                    "analyzers": [
                        {"name": "aruco_displacement"},
                    ],
                    "visualizers": [
                        {"name": "aruco_motion_plot", "result_indices": [0]},
                        {"name": "aruco_annotated_video", "result_indices": [0]},
                    ],
                }
            }
        )

        plan = Pipeline(context).execution_plan()

        self.assertTrue(plan.streamable_end_to_end)
        self.assertEqual(plan.materialization_boundaries, ())
        self.assertTrue(all(stage.execution_mode == "streaming" for stage in plan.stages))

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
        self.assertEqual(len(outputs.final_artifacts), 1)
        self.assertIsInstance(outputs.final_artifacts[0], VIDEO_ARTIFACT_TYPES)
        self.assertGreater(self._materialized_video_size(outputs.final_artifacts[0]), 0)

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
    def _generate_marker_image(marker_id: int, side: int, dictionary_id: int = cv2.aruco.DICT_6X6_250) -> np.ndarray:
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, "generateImageMarker"):
            return cv2.aruco.generateImageMarker(dictionary, marker_id, side)

        marker = np.zeros((side, side), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, side, marker, 1)
        return marker

    @staticmethod
    def _aruco_sample(frame_index: int, center_x: float, center_y: float, *, quality: float) -> ArucoMarkerSignalSample:
        half_side = 10.0
        corners = (
            (center_x - half_side, center_y - half_side),
            (center_x + half_side, center_y - half_side),
            (center_x + half_side, center_y + half_side),
            (center_x - half_side, center_y + half_side),
        )
        return ArucoMarkerSignalSample(
            frame_index=frame_index,
            markers=[
                ArucoMarkerObservation(
                    marker_id=7,
                    corners=corners,
                    center_x=center_x,
                    center_y=center_y,
                    detected=True,
                    quality_score=quality,
                    metadata={},
                )
            ],
            timestamp_seconds=frame_index * 0.1,
            metadata={},
        )

    @staticmethod
    def _center_span(points: list[tuple[float, float] | None]) -> float:
        valid_points = [point for point in points if point is not None]
        min_x = min(point[0] for point in valid_points)
        max_x = max(point[0] for point in valid_points)
        min_y = min(point[1] for point in valid_points)
        max_y = max(point[1] for point in valid_points)
        return max(max_x - min_x, max_y - min_y)

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
