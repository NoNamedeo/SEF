from __future__ import annotations

from typing import Any

import cv2

from library.core.enum.FrameRotation import FrameRotation
from library.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from library.core.artifacts.Frame import Frame


class OpenCVRotateFrameProcessor(ISingleFrameProcessor):
    """
    Rotates a frame by 90, 180, or 270 degrees.

    Rotation mapping:
        1 -> 90°
        2 -> 180°
        3 -> 270°
    """

    def __init__(
            self,
            rotation: FrameRotation = FrameRotation.ROTATE_90,
            config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.rotation = rotation

    def process(self, frame: Frame) -> Frame:
        image = frame.frame

        match self.rotation:
            case FrameRotation.ROTATE_90:
                rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            case FrameRotation.ROTATE_180:
                rotated = cv2.rotate(image, cv2.ROTATE_180)
            case FrameRotation.ROTATE_270:
                rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return Frame(
            image=rotated,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata={
                **dict(frame.metadata),
                "rotation": self.rotation,
            },
        )