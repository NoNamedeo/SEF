from __future__ import annotations

import unittest

import numpy as np

from sef.builtin.frame_processors.RealtimeFrameTapProcessor import RealtimeFrameTapProcessor
from sef.builtin.visualizers.COCO_pose.RealtimeCOCOPoseFrameVisualizer import RealtimeCOCOPoseFrameVisualizer
from sef.core.artifacts.data.COCOPoseTennisFrameData import COCOPoseTennisFrameData
from sef.core.realtime.LatestRealtimeFrameStore import LatestRealtimeFrameStore
from sef.core.realtime.RealtimeFrame import RealtimeFrame
from ui.services.realtime_preview_service import with_realtime_sink_ids
from ui.services.webcam_preflight_service import webcam_camera_index


class RealtimeVisualizationTests(unittest.TestCase):
    def test_latest_frame_store_keeps_copy_of_newest_frame(self) -> None:
        store = LatestRealtimeFrameStore()
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        store.publish(RealtimeFrame(image=image, frame_index=7))
        image[:, :] = 255

        snapshot = store.snapshot()

        self.assertEqual(snapshot.version, 1)
        self.assertTrue(snapshot.active)
        self.assertIsNotNone(snapshot.frame)
        self.assertEqual(snapshot.frame.frame_index, 7)
        self.assertEqual(int(snapshot.frame.image.sum()), 0)
        self.assertEqual(snapshot.published_frames, 1)

    def test_realtime_coco_visualizer_publishes_rendered_frame(self) -> None:
        store = LatestRealtimeFrameStore()
        visualizer = RealtimeCOCOPoseFrameVisualizer(
            config={"draw_source_frame": False, "canvas_size": (320, 240)},
            sink=store,
        )
        frame = _pose_frame()

        visualizer.render_stream((frame,))
        snapshot = store.snapshot()

        self.assertFalse(snapshot.active)
        self.assertIsNotNone(snapshot.frame)
        self.assertEqual(snapshot.frame.frame_index, 3)
        self.assertEqual(snapshot.frame.image.shape, (240, 320, 3))
        self.assertGreater(int(snapshot.frame.image.sum()), 0)

    def test_webcam_camera_index_reads_config(self) -> None:
        config = {
            "pipeline": {
                "frame_extractor": {
                    "name": "opencv_webcam",
                    "params": {"camera_index": 2, "config": {"max_frames": None}},
                }
            }
        }

        self.assertEqual(webcam_camera_index(config), 2)

    def test_realtime_frame_tap_publishes_raw_frame(self) -> None:
        store = LatestRealtimeFrameStore()
        processor = RealtimeFrameTapProcessor(sink=store)
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        processor.process(_frame_buffer_with(image))
        snapshot = store.snapshot()

        self.assertIsNotNone(snapshot.frame)
        self.assertEqual(snapshot.frame.frame_index, 5)
        self.assertEqual(snapshot.frame.metadata["preview_stage"], "frame_tap")
        self.assertEqual(snapshot.last_stage, "frame_tap")

    def test_annotated_frame_is_not_replaced_by_later_raw_tap(self) -> None:
        store = LatestRealtimeFrameStore()
        raw = np.zeros((8, 8, 3), dtype=np.uint8)
        annotated = np.full((8, 8, 3), 255, dtype=np.uint8)

        store.publish(
            RealtimeFrame(
                image=raw,
                frame_index=1,
                metadata={"preview_stage": "frame_tap", "preview_priority": 10},
            )
        )
        store.publish(
            RealtimeFrame(
                image=annotated,
                frame_index=1,
                metadata={"preview_stage": "coco_pose_visualizer", "preview_priority": 100},
            )
        )
        store.publish(
            RealtimeFrame(
                image=raw,
                frame_index=8,
                metadata={"preview_stage": "frame_tap", "preview_priority": 10},
            )
        )

        snapshot = store.snapshot()

        self.assertEqual(snapshot.frame.frame_index, 1)
        self.assertEqual(snapshot.last_stage, "coco_pose_visualizer")
        self.assertEqual(int(snapshot.frame.image.sum()), int(annotated.sum()))

    def test_realtime_sink_injection_adds_frame_tap(self) -> None:
        config = {
            "pipeline": {
                "frame_processors": [],
                "visualizers": [
                    {
                        "name": "streamlit_coco_pose_realtime",
                        "params": {"config": {}},
                        "result_indices": [0],
                    }
                ],
            }
        }

        patched = with_realtime_sink_ids(config, "preview-id")

        self.assertEqual(patched["pipeline"]["frame_processors"][0]["name"], "realtime_frame_tap")
        self.assertEqual(patched["pipeline"]["frame_processors"][0]["params"]["sink_id"], "preview-id")
        self.assertEqual(patched["pipeline"]["visualizers"][0]["params"]["sink_id"], "preview-id")


def _pose_frame() -> COCOPoseTennisFrameData:
    skeleton = np.zeros((17, 2), dtype=float)
    confidence = np.ones(17, dtype=float)
    for index in range(17):
        skeleton[index] = (40 + index * 8, 60 + index * 5)
    return COCOPoseTennisFrameData(
        frame_index=3,
        skeleton=skeleton,
        confidence=confidence,
        tennis_movement="ready",
        centroid=(120.0, 120.0),
        frame_size=(320, 240),
    )


def _frame_buffer_with(image: np.ndarray):
    from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
    from sef.core.artifacts.Frame import Frame

    buffer = FrameBuffer(buffer_size=2)
    buffer.put(Frame(image=image, index=5))
    buffer.close()
    return buffer


if __name__ == "__main__":
    unittest.main()
