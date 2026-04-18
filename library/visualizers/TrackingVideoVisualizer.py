from __future__ import annotations

import tempfile
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from library.core.artifacts.TrackingPlaybackData import TrackingPlaybackData
from library.core.interfaces.IData import IData
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.visualization.VisualArtifact import VideoArtifact
from library.core.visualization.VisualizationContext import VisualizationContext


class TrackingVideoVisualizer(IVisualizer):
    """Render tracking samples back into an annotated video artifact."""

    DEFAULT_FPS = 24.0
    CODEC_CANDIDATES = ("avc1", "H264", "mp4v")

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VideoArtifact, ...]:
        if not isinstance(data, TrackingPlaybackData):
            raise TypeError(
                "TrackingVideoVisualizer requires TrackingPlaybackData, "
                f"got {type(data).__name__}."
            )

        artifact_bytes, resolved_codec = self._render_video_bytes(data)
        return (
            VideoArtifact(
                kind="video",
                title=data.title,
                description="Tracked bounding boxes rendered on sampled frames.",
                metadata=self._artifact_metadata(
                    context,
                    extra={
                        "source_path": data.source_path,
                        "resize": data.resize,
                        "frame_count": len(data.frames),
                        "video_codec": resolved_codec,
                    },
                ),
                mime_type="video/mp4",
                data=artifact_bytes,
            ),
        )

    def _render_video_bytes(self, data: TrackingPlaybackData) -> tuple[bytes, str]:
        if not data.frames:
            raise ValueError("TrackingPlaybackData.frames cannot be empty.")
        if not data.source_path:
            raise ValueError("TrackingPlaybackData.source_path is required to render a tracking video.")

        source_path = Path(data.source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Tracking source video not found: {source_path}")

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video for tracking playback: {source_path}")

        ordered_frames = sorted(data.frames, key=lambda item: item.frame_index)
        output_size = self._resolve_output_size(capture, data)
        output_fps = data.fps or float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or self.DEFAULT_FPS

        file_descriptor, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(file_descriptor)
        output_path = Path(temp_path)
        writer, resolved_codec = self._open_video_writer(
            output_path=output_path,
            fps=output_fps,
            size=output_size,
        )
        if not writer.isOpened():
            capture.release()
            output_path.unlink(missing_ok=True)
            raise RuntimeError("Failed to initialize the MP4 writer for tracking playback.")

        try:
            for playback_frame in ordered_frames:
                image = self._read_frame(capture, playback_frame.frame_index)
                if data.resize is not None:
                    image = cv2.resize(image, data.resize)
                annotated = self._draw_tracks(image, playback_frame.tracks)
                writer.write(annotated)
        finally:
            writer.release()
            capture.release()

        try:
            return output_path.read_bytes(), resolved_codec
        finally:
            output_path.unlink(missing_ok=True)

    def _resolve_output_size(
        self,
        capture: cv2.VideoCapture,
        data: TrackingPlaybackData,
    ) -> tuple[int, int]:
        if data.resize is not None:
            return data.resize
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise ValueError("Cannot infer source frame size for tracking playback.")
        return width, height

    def _open_video_writer(
        self,
        *,
        output_path: Path,
        fps: float,
        size: tuple[int, int],
    ) -> tuple[cv2.VideoWriter, str]:
        for codec in self.CODEC_CANDIDATES:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                size,
            )
            if writer.isOpened():
                return writer, codec
            writer.release()
        return cv2.VideoWriter(), self.CODEC_CANDIDATES[-1]

    @staticmethod
    def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, image = capture.read()
        if not success or image is None:
            raise ValueError(f"Cannot read source frame {frame_index} for tracking playback.")
        return image

    def _draw_tracks(
        self,
        image: np.ndarray,
        tracks,
    ) -> np.ndarray:
        annotated = image.copy()
        for track in tracks:
            x, y, w, h = track.box
            color = self._track_color(track.track_id)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = track.label or f"ID {track.track_id}"
            cv2.putText(
                annotated,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            if track.centroid is not None:
                cx, cy = track.centroid
                cv2.circle(annotated, (int(cx), int(cy)), 4, color, -1)
        return annotated

    @staticmethod
    def _track_color(track_id: int) -> tuple[int, int, int]:
        palette = (
            (0, 255, 0),
            (255, 170, 0),
            (0, 200, 255),
            (255, 80, 80),
            (180, 80, 255),
            (120, 255, 120),
        )
        return palette[track_id % len(palette)]

    @staticmethod
    def _artifact_metadata(
        context: VisualizationContext | None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = dict(extra or {})
        if context is None:
            return metadata
        if context.pipeline_id is not None:
            metadata.setdefault("pipeline_id", context.pipeline_id)
        if context.analyzer_name is not None:
            metadata.setdefault("analyzer_name", context.analyzer_name)
        if context.visualizer_name is not None:
            metadata.setdefault("visualizer_name", context.visualizer_name)
        if context.result_index is not None:
            metadata.setdefault("result_index", context.result_index)
        if context.source_metadata:
            metadata.setdefault("source_metadata", dict(context.source_metadata))
        if context.execution_metadata:
            metadata.setdefault("execution_metadata", dict(context.execution_metadata))
        return metadata
