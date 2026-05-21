from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from library.core.artifacts.COCOPoseFrameData import COCOPoseFrameData, COCOPoseSequenceData
from library.core.interfaces.IData import IData
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import IStreamingVisualizer
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext


class OpenCVCOCOPoseRealtimeVisualizer(IStreamingVisualizer):
    """Render COCO pose keypoints in an OpenCV window while consuming stream data."""

    requires_main_thread = True

    capabilities = StageCapabilities.streaming(
        stateful=True,
        preserves_order=True,
        realtime_safe=True,
    )

    COCO_EDGES: Sequence[tuple[int, int]] = (
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    )

    def __init__(self, config=None):
        super().__init__(config)
        self.window_name = self.config.get("window_name", "YOLO Pose COCO")
        self.draw_source_frame = bool(self.config.get("draw_source_frame", True))
        self.keypoint_threshold = float(self.config.get("keypoint_threshold", 0.3))
        self.canvas_size = tuple(self.config.get("canvas_size", (1280, 720)))
        self.wait_ms = int(self.config.get("wait_ms", 1))
        self.joint_radius = int(self.config.get("joint_radius", 4))
        self.line_thickness = int(self.config.get("line_thickness", 2))
        self.background_color = tuple(self.config.get("background_color", (24, 24, 24)))
        self.joint_color = tuple(self.config.get("joint_color", (0, 255, 0)))
        self.edge_color = tuple(self.config.get("edge_color", (255, 0, 0)))
        self.centroid_color = tuple(self.config.get("centroid_color", (0, 0, 255)))

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        if isinstance(data, COCOPoseSequenceData):
            return self.render_stream(data.frames, context)
        if isinstance(data, COCOPoseFrameData):
            return self.render_stream((data,), context)
        raise TypeError(
            "OpenCVCOCOPoseRealtimeVisualizer requires COCOPoseFrameData "
            f"or COCOPoseSequenceData, got {type(data).__name__}."
        )

    def render_stream(
        self,
        data: Iterable[IData],
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        display_enabled = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        try:
            for item in data:
                pose_frame = self._require_pose_frame(item)
                if not display_enabled:
                    continue

                canvas = self._build_canvas(pose_frame)
                self._draw_pose(canvas, pose_frame)
                self._draw_status(canvas, pose_frame)
                cv2.imshow(self.window_name, np.ascontiguousarray(canvas))

                if self._should_stop(cv2.waitKey(self.wait_ms)):
                    self._abort_stream(data)
                    display_enabled = False
                    cv2.destroyWindow(self.window_name)
                    break
        finally:
            if display_enabled:
                cv2.destroyWindow(self.window_name)

        return ()

    def _build_canvas(self, pose_frame: COCOPoseFrameData) -> np.ndarray:
        if self.draw_source_frame and pose_frame.frame_image is not None:
            return pose_frame.frame_image.copy()

        width, height = pose_frame.frame_size or self.canvas_size
        return np.full((int(height), int(width), 3), self.background_color, dtype=np.uint8)

    def _draw_pose(self, canvas: np.ndarray, pose_frame: COCOPoseFrameData) -> None:
        skeleton = pose_frame.skeleton
        confidence = pose_frame.confidence
        if skeleton is None or confidence is None:
            return

        for start_index, end_index in self.COCO_EDGES:
            if not self._is_visible(confidence, start_index) or not self._is_visible(confidence, end_index):
                continue
            cv2.line(
                canvas,
                self._point(skeleton[start_index]),
                self._point(skeleton[end_index]),
                self.edge_color,
                self.line_thickness,
                cv2.LINE_AA,
            )

        for keypoint_index, keypoint in enumerate(skeleton):
            if not self._is_visible(confidence, keypoint_index):
                continue
            cv2.circle(canvas, self._point(keypoint), self.joint_radius, self.joint_color, -1, cv2.LINE_AA)

        if (
            pose_frame.centroid is not None
            and self._is_visible(confidence, 11)
            and self._is_visible(confidence, 12)
        ):
            cv2.circle(canvas, self._point(np.asarray(pose_frame.centroid)), 5, self.centroid_color, -1, cv2.LINE_AA)

    def _draw_status(self, canvas: np.ndarray, pose_frame: COCOPoseFrameData) -> None:
        label = f"frame: {pose_frame.frame_index} | ESC/q: close preview"
        cv2.putText(
            canvas,
            label,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _is_visible(self, confidence: np.ndarray, keypoint_index: int) -> bool:
        return bool(confidence[keypoint_index] >= self.keypoint_threshold)

    @staticmethod
    def _point(point: np.ndarray) -> tuple[int, int]:
        return int(round(float(point[0]))), int(round(float(point[1])))

    @staticmethod
    def _should_stop(key_code: int) -> bool:
        normalized_key = key_code & 0xFF
        return normalized_key in (27, ord("q"), ord("Q"))

    @staticmethod
    def _abort_stream(data: Iterable[IData]) -> None:
        abort = getattr(data, "abort", None)
        if callable(abort):
            abort()

    @staticmethod
    def _require_pose_frame(item: IData) -> COCOPoseFrameData:
        if not isinstance(item, COCOPoseFrameData):
            raise TypeError(
                "OpenCVCOCOPoseRealtimeVisualizer stream requires COCOPoseFrameData, "
                f"got {type(item).__name__}."
            )
        return item
