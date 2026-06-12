from __future__ import annotations

from typing import Any

import cv2

from sef.core.artifacts.Frame import Frame
from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.interfaces.BufferContracts import IFrameBuffer
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.interfaces.StreamingContracts import IStreamingFrameExtractor
from sef.core.pipeline.LatencyPolicy import BlockingFrameLatencyPolicy, FrameLatencyPolicy


class OpenCVWebcamFrameExtractor(IStreamingFrameExtractor):
    """Capture frames from a local webcam into a bounded streaming buffer."""

    capabilities = StageCapabilities.streaming(
        stateful=True,
        preserves_order=True,
        realtime_safe=True,
    )

    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_MAX_FRAMES = 300
    DEFAULT_STREAM_BUFFER_SIZE = 4

    def __init__(
        self,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.camera_index = int(camera_index)
        self.resize = self.config.get("resize")
        self.mirror = bool(self.config.get("mirror", False))
        self.stride = self._optional_positive_int(self.config.get("stride", 1), field_name="stride") or 1
        self.max_frames = self._optional_positive_int(
            self.config.get("max_frames", self.DEFAULT_MAX_FRAMES),
            field_name="max_frames",
        )
        self.requested_width = self._optional_positive_int(self.config.get("width"), field_name="width")
        self.requested_height = self._optional_positive_int(self.config.get("height"), field_name="height")
        self.requested_fps = self._optional_positive_float(self.config.get("fps"), field_name="fps")

    def extract(self) -> FrameBuffer:
        """Capture a bounded sequence for non-streaming callers."""
        capacity = self.max_frames + 1 if self.max_frames is not None else self.DEFAULT_MAX_FRAMES + 1
        buffer = FrameBuffer(buffer_size=capacity)
        self.extract_into(buffer, BlockingFrameLatencyPolicy())
        return buffer

    def extract_into(
        self,
        output_buffer: IFrameBuffer,
        latency_policy: FrameLatencyPolicy,
    ) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise ValueError(f"Cannot open webcam index {self.camera_index}.")

        self._apply_capture_properties(capture)
        frame_index = 0
        try:
            captured_frames = 0
            while self.max_frames is None or frame_index < self.max_frames:
                if output_buffer.closed:
                    break

                success, image = capture.read()
                if not success:
                    break
                if captured_frames % self.stride != 0:
                    captured_frames += 1
                    continue
                if self.resize is not None:
                    image = cv2.resize(image, self.resize)
                if self.mirror:
                    image = cv2.flip(image, 1)

                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                timestamp_seconds = (frame_index / fps) if fps > 0 else None
                frame = Frame(
                    image=image,
                    index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    metadata={
                        "source": "webcam",
                        "camera_index": self.camera_index,
                        "frame_size": (int(image.shape[1]), int(image.shape[0])),
                        **({"source_fps": fps} if fps > 0 else {}),
                    },
                )
                latency_policy.publish(frame, output_buffer)
                frame_index += 1
                captured_frames += 1
        finally:
            capture.release()
            output_buffer.close()

    def _apply_capture_properties(self, capture: cv2.VideoCapture) -> None:
        if self.requested_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        if self.requested_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        if self.requested_fps is not None:
            capture.set(cv2.CAP_PROP_FPS, self.requested_fps)

    @staticmethod
    def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
        if value is None:
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        return parsed

    @staticmethod
    def _optional_positive_float(value: Any, *, field_name: str) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        if parsed <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        return parsed
