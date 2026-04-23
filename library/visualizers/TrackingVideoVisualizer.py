from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Protocol

import cv2
import numpy as np

from library.core.artifacts.TrackingPlaybackData import TrackingPlaybackData
from library.core.interfaces.IData import IData
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.visualization.VisualArtifact import (
    DeferredVideoArtifact,
    VideoFileArtifact,
    VideoLikeArtifact,
)
from library.core.visualization.VisualizationContext import VisualizationContext


class _VideoFrameWriter(Protocol):
    """Minimal sink interface for encoded video writers."""

    def write(self, frame: np.ndarray) -> None:
        """Append one BGR frame to the output stream."""

    def close(self) -> None:
        """Flush and close the encoded output."""


@dataclass(frozen=True, slots=True)
class VideoRenderOptions:
    """
    Configurable annotated-video rendering options.

    The defaults favor low runtime and low RAM: rendering is lazy, frames are
    streamed from the source video, and encoded output is file-backed.
    """

    lazy: bool = True
    frame_sample_interval: int = 1
    downscale_factor: float | None = None
    output_size: tuple[int, int] | None = None
    max_width: int | None = None
    max_height: int | None = None
    output_fps: float | None = None
    preserve_timing: bool = True
    encoder_backend: str = "auto"
    codec: str | None = None
    codec_candidates: tuple[str, ...] = ("avc1", "H264", "mp4v")
    preset: str | None = None
    output_directory: Path | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> VideoRenderOptions:
        """Parse public visualizer config with validation and aliases."""
        sample_interval = cls._positive_int(
            cls._first_present(
                config,
                "frame_sample_interval",
                "sample_every_n_frames",
                "sampling_stride",
                "frame_stride",
            ),
            default=1,
            field_name="frame_sample_interval",
        )
        preset = cls._first_present(config, "preset", "ffmpeg_preset")
        return cls(
            lazy=bool(config.get("lazy", True)),
            frame_sample_interval=sample_interval,
            downscale_factor=cls._optional_positive_float(
                cls._first_present(config, "downscale_factor", "downscale", "scale"),
                field_name="downscale_factor",
            ),
            output_size=cls._optional_size(config.get("output_size")),
            max_width=cls._optional_positive_int(config.get("max_width"), field_name="max_width"),
            max_height=cls._optional_positive_int(config.get("max_height"), field_name="max_height"),
            output_fps=cls._optional_positive_float(config.get("output_fps"), field_name="output_fps"),
            preserve_timing=bool(config.get("preserve_timing", True)),
            encoder_backend=str(config.get("encoder_backend", "auto")).lower(),
            codec=str(config["codec"]) if config.get("codec") else None,
            codec_candidates=cls._codec_candidates(config),
            preset=str(preset) if preset is not None else None,
            output_directory=cls._optional_path(
                cls._first_present(config, "output_directory", "output_dir"),
            ),
        )

    def metadata(self) -> dict[str, Any]:
        """Return JSON-friendly render settings for artifact metadata."""
        metadata: dict[str, Any] = {
            "lazy": self.lazy,
            "frame_sample_interval": self.frame_sample_interval,
            "preserve_timing": self.preserve_timing,
            "encoder_backend": self.encoder_backend,
        }
        if self.downscale_factor is not None:
            metadata["downscale_factor"] = self.downscale_factor
        if self.output_size is not None:
            metadata["output_size"] = self.output_size
        if self.max_width is not None:
            metadata["max_width"] = self.max_width
        if self.max_height is not None:
            metadata["max_height"] = self.max_height
        if self.output_fps is not None:
            metadata["output_fps"] = self.output_fps
        if self.codec is not None:
            metadata["codec"] = self.codec
        if self.preset is not None:
            metadata["preset"] = self.preset
        if self.output_directory is not None:
            metadata["output_directory"] = str(self.output_directory)
        return metadata

    @staticmethod
    def _first_present(config: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in config:
                return config[key]
        return None

    @staticmethod
    def _positive_int(value: Any, *, default: int, field_name: str) -> int:
        if value is None:
            return default
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        return parsed

    @staticmethod
    def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
        if value is None:
            return None
        return VideoRenderOptions._positive_int(value, default=1, field_name=field_name)

    @staticmethod
    def _optional_positive_float(value: Any, *, field_name: str) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        if parsed <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        return parsed

    @staticmethod
    def _optional_size(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError("output_size must be a two-item sequence: [width, height].")
        width = int(value[0])
        height = int(value[1])
        if width <= 0 or height <= 0:
            raise ValueError("output_size width and height must be greater than 0.")
        return width, height

    @staticmethod
    def _optional_path(value: Any) -> Path | None:
        if value is None:
            return None
        return Path(str(value))

    @staticmethod
    def _codec_candidates(config: Mapping[str, Any]) -> tuple[str, ...]:
        codec = config.get("codec")
        raw_candidates = config.get("codec_candidates")
        candidates: list[str] = []
        if codec:
            candidates.append(str(codec))
        if raw_candidates is not None:
            if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
                raise ValueError("codec_candidates must be a sequence of codec strings.")
            candidates.extend(str(item) for item in raw_candidates)
        candidates.extend(("avc1", "H264", "mp4v"))
        return tuple(OrderedDict.fromkeys(candidates))


class TrackingVideoVisualizer(IVisualizer):
    """Return lazy, file-backed annotated video artifacts for tracking samples."""

    DEFAULT_FPS = 24.0
    DEFAULT_MIME_TYPE = "video/mp4"
    DEFAULT_SUFFIX = ".mp4"

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VideoLikeArtifact, ...]:
        if not isinstance(data, TrackingPlaybackData):
            raise TypeError(f"TrackingVideoVisualizer requires TrackingPlaybackData, got {type(data).__name__}.")

        return self._build_video_artifact(data, context)

    def _build_video_artifact(
        self,
        data: Any,
        context: VisualizationContext | None,
    ) -> tuple[VideoLikeArtifact, ...]:
        options = VideoRenderOptions.from_config(self.config)
        frames = self._selected_annotation_frames(data, options)
        metadata = self._artifact_metadata(
            context,
            extra={
                "source_path": self._source_path(data),
                "resize": self._analysis_resize(data),
                "frame_count": len(self._annotation_frames(data)),
                "rendered_frame_count": len(frames),
                "render_options": options.metadata(),
            },
        )

        if options.lazy:
            return (
                DeferredVideoArtifact(
                    kind="video",
                    title=self._artifact_title(data),
                    description=self._artifact_description(),
                    metadata=metadata,
                    mime_type=self.DEFAULT_MIME_TYPE,
                    filename_suffix=self.DEFAULT_SUFFIX,
                    render_to=lambda output_path: self._render_video_file(data, output_path, options),
                ),
            )

        output_path = self._new_output_path(options)
        self._render_video_file(data, output_path, options)
        return (
            VideoFileArtifact(
                kind="video",
                title=self._artifact_title(data),
                description=self._artifact_description(),
                metadata=metadata,
                mime_type=self.DEFAULT_MIME_TYPE,
                path=output_path,
            ),
        )

    def _render_video_file(
        self,
        data: Any,
        output_path: Path,
        options: VideoRenderOptions,
    ) -> Path:
        frames = self._selected_annotation_frames(data, options)
        if not frames:
            raise ValueError("Annotated video rendering requires at least one selected frame.")

        raw_source_path = self._source_path(data)
        if not raw_source_path:
            raise ValueError(f"{type(data).__name__}.source_path is required to render an annotated video.")

        source_path = Path(raw_source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Tracking source video not found: {source_path}")

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video for tracking playback: {source_path}")

        writer: _VideoFrameWriter | None = None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        try:
            analysis_size = self._resolve_analysis_size(capture, data)
            output_size = self._resolve_output_size(analysis_size, options)
            source_fps = self._resolve_source_fps(capture, data)
            output_fps = self._resolve_output_fps(frames, source_fps, options)
            writer = self._open_video_writer(
                output_path=output_path,
                fps=output_fps,
                size=output_size,
                options=options,
            )
            annotation_context = self._annotation_context(data)

            written_frames = 0
            for annotation_frame, image in self._read_selected_frames(capture, frames):
                prepared = self._prepare_frame(image, analysis_size)
                annotated = self._draw_annotation_frame(prepared, annotation_frame, data, annotation_context)
                if output_size != analysis_size:
                    annotated = cv2.resize(annotated, output_size)
                writer.write(annotated)
                written_frames += 1

            if written_frames == 0:
                raise ValueError("Annotated video rendering did not write any frames.")
        finally:
            try:
                if writer is not None:
                    writer.close()
            finally:
                capture.release()

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"Annotated video writer did not produce a valid file: {output_path}")
        return output_path

    def _annotation_frames(self, data: TrackingPlaybackData) -> list[Any]:
        """Return annotation frames sorted by source frame index and deduplicated."""
        deduplicated: OrderedDict[int, Any] = OrderedDict()
        for frame in sorted(data.frames, key=lambda item: int(item.frame_index)):
            deduplicated[int(frame.frame_index)] = frame
        return list(deduplicated.values())

    def _selected_annotation_frames(
        self,
        data: Any,
        options: VideoRenderOptions,
    ) -> list[Any]:
        frames = self._annotation_frames(data)
        if not frames:
            raise ValueError(f"{type(data).__name__}.frames cannot be empty.")
        first_frame_index = self._annotation_frame_index(frames[0])
        return [frame for frame in frames if (self._annotation_frame_index(frame) - first_frame_index) % options.frame_sample_interval == 0]

    def _read_selected_frames(
        self,
        capture: cv2.VideoCapture,
        annotation_frames: Sequence[Any],
    ):
        current_frame_index = -1
        for annotation_frame in annotation_frames:
            target_frame_index = self._annotation_frame_index(annotation_frame)
            if target_frame_index < current_frame_index:
                raise ValueError("Annotation frames must be sorted by ascending frame_index.")
            while current_frame_index < target_frame_index:
                success = capture.grab()
                if not success:
                    raise ValueError(f"Cannot read source frame {target_frame_index} for tracking playback.")
                current_frame_index += 1
            success, image = capture.retrieve()
            if not success or image is None:
                raise ValueError(f"Cannot retrieve source frame {target_frame_index} for tracking playback.")
            yield annotation_frame, image

    def _prepare_frame(self, image: np.ndarray, analysis_size: tuple[int, int]) -> np.ndarray:
        if (int(image.shape[1]), int(image.shape[0])) == analysis_size:
            return image
        return cv2.resize(image, analysis_size)

    def _annotation_context(self, data: Any) -> Any:
        """Build per-render annotation state once before frame streaming starts."""
        return None

    def _draw_annotation_frame(
        self,
        image: np.ndarray,
        annotation_frame: Any,
        data: Any,
        annotation_context: Any,
    ) -> np.ndarray:
        return self._draw_tracks(image, annotation_frame.tracks)

    def _artifact_title(self, data: Any) -> str:
        return data.title

    def _artifact_description(self) -> str:
        return "Tracked bounding boxes rendered on sampled frames."

    def _source_path(self, data: Any) -> str | None:
        return data.source_path

    def _analysis_resize(self, data: Any) -> tuple[int, int] | None:
        return data.resize

    def _source_fps(self, data: Any) -> float | None:
        return data.fps

    @staticmethod
    def _annotation_frame_index(annotation_frame: Any) -> int:
        return int(annotation_frame.frame_index)

    def _resolve_analysis_size(
        self,
        capture: cv2.VideoCapture,
        data: Any,
    ) -> tuple[int, int]:
        resize = self._analysis_resize(data)
        if resize is not None:
            return resize
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise ValueError("Cannot infer source frame size for tracking playback.")
        return width, height

    def _resolve_output_size(
        self,
        analysis_size: tuple[int, int],
        options: VideoRenderOptions,
    ) -> tuple[int, int]:
        if options.output_size is not None:
            return self._even_size(options.output_size)

        width, height = analysis_size
        scale = 1.0
        if options.downscale_factor is not None:
            scale *= options.downscale_factor

        if options.max_width is not None and width * scale > options.max_width:
            scale *= options.max_width / (width * scale)
        if options.max_height is not None and height * scale > options.max_height:
            scale *= options.max_height / (height * scale)

        return self._even_size((max(1, round(width * scale)), max(1, round(height * scale))))

    def _resolve_source_fps(self, capture: cv2.VideoCapture, data: Any) -> float:
        return self._source_fps(data) or float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or self.DEFAULT_FPS

    def _resolve_output_fps(
        self,
        frames: Sequence[Any],
        source_fps: float,
        options: VideoRenderOptions,
    ) -> float:
        if options.output_fps is not None:
            return options.output_fps
        if not options.preserve_timing or len(frames) < 2:
            return source_fps

        frame_gaps = [
            self._annotation_frame_index(current) - self._annotation_frame_index(previous) for previous, current in zip(frames, frames[1:])
        ]
        positive_gaps = [gap for gap in frame_gaps if gap > 0]
        if not positive_gaps:
            return source_fps
        return max(0.01, source_fps / float(median(positive_gaps)))

    def _open_video_writer(
        self,
        *,
        output_path: Path,
        fps: float,
        size: tuple[int, int],
        options: VideoRenderOptions,
    ) -> _VideoFrameWriter:
        backend = self._resolve_encoder_backend(options)
        if backend == "ffmpeg":
            return _FFmpegVideoFrameWriter.open(
                output_path=output_path,
                fps=fps,
                size=size,
                codec=options.codec or "libx264",
                preset=options.preset or "ultrafast",
            )
        return _OpenCVVideoFrameWriter.open(
            output_path=output_path,
            fps=fps,
            size=size,
            codec_candidates=options.codec_candidates,
        )

    @staticmethod
    def _resolve_encoder_backend(options: VideoRenderOptions) -> str:
        backend = options.encoder_backend
        if backend not in {"auto", "opencv", "ffmpeg"}:
            raise ValueError("encoder_backend must be one of: auto, opencv, ffmpeg.")

        ffmpeg_available = shutil.which("ffmpeg") is not None
        if backend == "ffmpeg":
            if not ffmpeg_available:
                raise RuntimeError("encoder_backend='ffmpeg' requires ffmpeg to be installed.")
            return "ffmpeg"
        if backend == "opencv":
            return "opencv"
        if ffmpeg_available and (options.preset is not None or (options.codec is not None and len(options.codec) != 4)):
            return "ffmpeg"
        return "opencv"

    @staticmethod
    def _new_output_path(options: VideoRenderOptions) -> Path:
        output_directory = options.output_directory or Path(tempfile.gettempdir()) / "sef_video_artifacts"
        output_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_directory, suffix=".mp4", delete=False) as temp_file:
            return Path(temp_file.name)

    @staticmethod
    def _even_size(size: tuple[int, int]) -> tuple[int, int]:
        width, height = size
        return max(2, int(width) - int(width) % 2), max(2, int(height) - int(height) % 2)

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


class _OpenCVVideoFrameWriter:
    """OpenCV VideoWriter adapter with explicit codec fallback."""

    def __init__(self, writer: cv2.VideoWriter) -> None:
        self._writer = writer

    @classmethod
    def open(
        cls,
        *,
        output_path: Path,
        fps: float,
        size: tuple[int, int],
        codec_candidates: Sequence[str],
    ) -> _OpenCVVideoFrameWriter:
        for codec in codec_candidates:
            if len(codec) != 4:
                continue
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                size,
            )
            if writer.isOpened():
                return cls(writer)
            writer.release()
        raise RuntimeError(f"Failed to initialize OpenCV MP4 writer with codecs: {tuple(codec_candidates)}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


class _FFmpegVideoFrameWriter:
    """ffmpeg raw-frame stdin writer used when codec presets are requested."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    @classmethod
    def open(
        cls,
        *,
        output_path: Path,
        fps: float,
        size: tuple[int, int],
        codec: str,
        preset: str,
    ) -> _FFmpegVideoFrameWriter:
        width, height = size
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            codec,
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return cls(process)

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available.")
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr = self._process.stderr.read() if self._process.stderr is not None else b""
        self._process.wait()
        if self._process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg video encoding failed: {message}")
