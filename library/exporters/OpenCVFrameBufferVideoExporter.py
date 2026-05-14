from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.interfaces.IFrameExporter import FrameExportContext, FrameExportResult, IFrameExporter
from library.core.visualization.VisualArtifact import ArtifactRole, VideoFileArtifact


class OpenCVFrameBufferVideoExporter(IFrameExporter):
    """
    Export a processed frame stream to an MP4 file-backed final artifact.

    The exporter is intentionally outside the frame-processing chain: it owns
    persistence and artifact creation, while processors stay pure transformations.
    It rebuilds a pass-through buffer so downstream signal extractors can still
    consume the same processed frames.
    """

    DEFAULT_CODEC = "mp4v"
    DEFAULT_MIME_TYPE = "video/mp4"
    DEFAULT_MAX_EXPORTED_FRAMES = 10_000

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        *,
        title: str = "Processed video",
        description: str = "Final processed video exported from the frame pipeline.",
        codec: str = DEFAULT_CODEC,
        max_exported_frames: int = DEFAULT_MAX_EXPORTED_FRAMES,
        config: dict[str, Any] | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than 0.")
        if len(codec) != 4:
            raise ValueError("codec must be a four-character OpenCV codec.")
        if max_exported_frames <= 0:
            raise ValueError("max_exported_frames must be greater than 0.")

        self.output_path = Path(output_path)
        self.fps = float(fps)
        self.title = title
        self.description = description
        self.codec = codec
        self.max_exported_frames = int(max_exported_frames)
        self.config = dict(config or {})

    def export(self, buffer: FrameBuffer, context: FrameExportContext) -> FrameExportResult:
        """Write the processed stream to disk and return a replayable buffer."""
        output_buffer = buffer.clone_empty()
        artifacts = self.export_into(buffer, output_buffer, context)
        return FrameExportResult(buffer=output_buffer, artifacts=artifacts)

    def export_into(
        self,
        frames: FrameBuffer,
        output_buffer: FrameBuffer,
        context: FrameExportContext,
    ) -> tuple[VideoFileArtifact, ...]:
        """Write frames to disk while forwarding them to the next streaming stage."""
        writer: cv2.VideoWriter | None = None
        expected_size: tuple[int, int] | None = None
        written_frames = 0

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()

        try:
            for frame in frames:
                if written_frames >= self.max_exported_frames:
                    raise ValueError(f"Video export exceeded max_exported_frames={self.max_exported_frames}.")

                size = self._frame_size(frame)
                if expected_size is None:
                    expected_size = size
                    writer = self._open_writer(expected_size)
                elif size != expected_size:
                    raise ValueError(f"All exported frames must have the same size. Expected {expected_size}, got {size}.")

                if writer is None:
                    raise RuntimeError("Video writer was not initialized.")

                writer.write(frame.image)
                output_buffer.put(frame)
                written_frames += 1
        finally:
            if writer is not None:
                writer.release()
            output_buffer.close()

        if written_frames == 0:
            raise ValueError("Cannot export a video from an empty frame buffer.")
        if not self.output_path.exists() or self.output_path.stat().st_size <= 0:
            raise RuntimeError(f"Video export did not produce a valid file: {self.output_path}")

        return (
                VideoFileArtifact(
                    kind="video",
                    role=ArtifactRole.FINAL_OUTPUT,
                    title=self.title,
                    description=self.description,
                    metadata={
                        "pipeline_id": context.pipeline_id,
                        "exporter_name": context.exporter_name,
                        "frame_count": written_frames,
                        "fps": self.fps,
                        "codec": self.codec,
                        "path": str(self.output_path),
                    },
                    mime_type=self.DEFAULT_MIME_TYPE,
                    path=self.output_path,
                ),
        )

    def _open_writer(self, size: tuple[int, int]) -> cv2.VideoWriter:
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            size,
        )
        if not writer.isOpened():
            raise ValueError(f"Cannot create output video: {self.output_path}")
        return writer

    @staticmethod
    def _frame_size(frame: Frame) -> tuple[int, int]:
        height, width = frame.image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be greater than 0.")
        return int(width), int(height)
