from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class COCOPoseRenderConfig:
    """Configuration for rendering COCO skeleton frames without choosing an output surface."""

    draw_source_frame: bool = True
    keypoint_threshold: float = 0.3
    canvas_size: tuple[int, int] = (1280, 720)
    joint_radius: int = 4
    line_thickness: int = 2
    background_color: tuple[int, int, int] = (24, 24, 24)
    joint_color: tuple[int, int, int] = (0, 255, 0)
    edge_color: tuple[int, int, int] = (255, 0, 0)
    centroid_color: tuple[int, int, int] = (0, 0, 255)
    label_color: tuple[int, int, int] = (255, 255, 255)
    label_shadow_color: tuple[int, int, int] = (20, 20, 20)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "COCOPoseRenderConfig":
        """Build a typed render config from user-facing plugin params."""
        if value is None:
            return cls()
        return cls(
            draw_source_frame=bool(value.get("draw_source_frame", True)),
            keypoint_threshold=float(value.get("keypoint_threshold", 0.3)),
            canvas_size=_tuple_int_pair(value.get("canvas_size", (1280, 720))),
            joint_radius=int(value.get("joint_radius", 4)),
            line_thickness=int(value.get("line_thickness", 2)),
            background_color=_color(value.get("background_color", (24, 24, 24))),
            joint_color=_color(value.get("joint_color", (0, 255, 0))),
            edge_color=_color(value.get("edge_color", (255, 0, 0))),
            centroid_color=_color(value.get("centroid_color", (0, 0, 255))),
            label_color=_color(value.get("label_color", (255, 255, 255))),
            label_shadow_color=_color(value.get("label_shadow_color", (20, 20, 20))),
        )


class COCOPoseFrameRenderer:
    """
    Stateless renderer for COCO pose-like frame objects.

    It intentionally depends on attribute names instead of concrete data classes
    so the same renderer can serve tennis-specific and generic COCO pose frames.
    """

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

    def __init__(self, config: COCOPoseRenderConfig | None = None) -> None:
        self._config = config or COCOPoseRenderConfig()

    def render(self, pose_frame: object) -> np.ndarray:
        """Render a BGR frame with skeleton, keypoints and status overlay."""
        canvas = self._build_canvas(pose_frame)
        self._draw_pose(canvas, pose_frame)
        self._draw_status(canvas, pose_frame)
        return np.ascontiguousarray(canvas)

    def _build_canvas(self, pose_frame: object) -> np.ndarray:
        frame_image = getattr(pose_frame, "frame_image", None)
        if self._config.draw_source_frame and isinstance(frame_image, np.ndarray):
            return frame_image.copy()

        frame_size = getattr(pose_frame, "frame_size", None)
        width, height = frame_size or self._config.canvas_size
        return np.full((int(height), int(width), 3), self._config.background_color, dtype=np.uint8)

    def _draw_pose(self, canvas: np.ndarray, pose_frame: object) -> None:
        skeleton = getattr(pose_frame, "skeleton", None)
        confidence = getattr(pose_frame, "confidence", None)
        if skeleton is None or confidence is None:
            return

        for start_index, end_index in self.COCO_EDGES:
            if not self._is_visible(confidence, start_index) or not self._is_visible(confidence, end_index):
                continue
            cv2.line(
                canvas,
                self._point(skeleton[start_index]),
                self._point(skeleton[end_index]),
                self._config.edge_color,
                self._config.line_thickness,
                cv2.LINE_AA,
            )

        for keypoint_index, keypoint in enumerate(skeleton):
            if not self._is_visible(confidence, keypoint_index):
                continue
            cv2.circle(
                canvas,
                self._point(keypoint),
                self._config.joint_radius,
                self._config.joint_color,
                -1,
                cv2.LINE_AA,
            )

        centroid = getattr(pose_frame, "centroid", None)
        if centroid is not None and self._is_visible(confidence, 11) and self._is_visible(confidence, 12):
            cv2.circle(canvas, self._point(np.asarray(centroid)), 5, self._config.centroid_color, -1, cv2.LINE_AA)

    def _draw_status(self, canvas: np.ndarray, pose_frame: object) -> None:
        frame_index = getattr(pose_frame, "frame_index", "?")
        movement = getattr(pose_frame, "tennis_movement", None)
        movement_suffix = f" | movement: {movement}" if movement is not None else ""
        label = f"frame: {frame_index}{movement_suffix}"

        cv2.putText(
            canvas,
            label,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            self._config.label_shadow_color,
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            self._config.label_color,
            2,
            cv2.LINE_AA,
        )

    def _is_visible(self, confidence: np.ndarray, keypoint_index: int) -> bool:
        return bool(confidence[keypoint_index] >= self._config.keypoint_threshold)

    @staticmethod
    def _point(point: np.ndarray) -> tuple[int, int]:
        return int(round(float(point[0]))), int(round(float(point[1])))


def _tuple_int_pair(value: object) -> tuple[int, int]:
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("Expected a pair of integers.")
    return int(value[0]), int(value[1])


def _color(value: object) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError("Expected a BGR color tuple with three integers.")
    return int(value[0]), int(value[1]), int(value[2])
