from __future__ import annotations

import tempfile
import threading
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

MAX_IMAGE_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_VIDEO_ARTIFACT_BYTES = 25 * 1024 * 1024


class ArtifactRole(StrEnum):
    """
    Semantic role used by UIs and exporters to place artifacts correctly.

    Roles are presentation hints. They do not change artifact data semantics.
    """

    FINAL_OUTPUT = "final_output"
    ANALYSIS = "analysis"
    DEBUG = "debug"
    PREVIEW = "preview"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualArtifact(ABC):
    """
    Base contract for UI-agnostic presentation artifacts.

    Visual artifacts are immutable values returned by visualizers and exporters.
    They describe what should be rendered or persisted without assuming a
    specific UI framework.

    Attributes
    ----------
    artifact_id:
        Stable id for UI diffing, downloads, and materialization caching.
    kind:
        Artifact kind string such as `image`, `video`, `table`, `json`, or
        `text`.
    role:
        Semantic placement hint for adapters.
    title:
        Optional display title.
    description:
        Optional explanatory text.
    metadata:
        JSON-like metadata for adapters and diagnostics.
    """

    artifact_id: str = field(default_factory=lambda: uuid4().hex)
    kind: str
    role: ArtifactRole | str = ArtifactRole.FINAL_OUTPUT
    title: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("VisualArtifact.kind must be a non-empty string.")
        object.__setattr__(self, "role", ArtifactRole(str(self.role)))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageArtifact(VisualArtifact):
    """
    In-memory image artifact ready for rendering or persistence.

    Use this for small images. Large images should generally be written through
    a file-backed or deferred artifact strategy.
    """

    mime_type: str
    data: bytes

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.mime_type:
            raise ValueError("ImageArtifact.mime_type must be a non-empty string.")
        if not self.data:
            raise ValueError("ImageArtifact.data cannot be empty.")
        if len(self.data) > MAX_IMAGE_ARTIFACT_BYTES:
            raise ValueError(
                f"ImageArtifact.data exceeds the hard in-memory limit of {MAX_IMAGE_ARTIFACT_BYTES} bytes."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoArtifact(VisualArtifact):
    """
    In-memory video artifact ready for UI playback or persistence.

    This value enforces a hard byte limit to protect UI and API adapters from
    accidentally carrying large encoded videos in memory.
    """

    mime_type: str
    data: bytes

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.mime_type:
            raise ValueError("VideoArtifact.mime_type must be a non-empty string.")
        if not self.data:
            raise ValueError("VideoArtifact.data cannot be empty.")
        if len(self.data) > MAX_VIDEO_ARTIFACT_BYTES:
            raise ValueError(
                f"VideoArtifact.data exceeds the hard in-memory limit of {MAX_VIDEO_ARTIFACT_BYTES} bytes. "
                "Use VideoFileArtifact or DeferredVideoArtifact for large videos."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoFileArtifact(VisualArtifact):
    """
    File-backed video artifact that avoids keeping encoded bytes in memory.

    The path must exist at construction time and point to a non-empty file.
    """

    mime_type: str
    path: Path | str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.mime_type:
            raise ValueError("VideoFileArtifact.mime_type must be a non-empty string.")
        path = Path(self.path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"VideoFileArtifact.path must point to an existing file: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"VideoFileArtifact.path cannot be empty: {path}")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeferredVideoArtifact(VisualArtifact):
    """
    Lazy video artifact that renders to disk only when a consumer materializes it.

    The render callback receives the target output path and must write a complete
    video file there. This keeps expensive annotated-video generation outside the
    main pipeline execution path and avoids in-memory MP4 buffering.
    """

    mime_type: str
    render_to: Callable[[Path], Path | VideoFileArtifact]
    filename_suffix: str = ".mp4"
    _materialized_path: Path | None = field(default=None, init=False, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.mime_type:
            raise ValueError("DeferredVideoArtifact.mime_type must be a non-empty string.")
        if not self.filename_suffix.startswith("."):
            raise ValueError("DeferredVideoArtifact.filename_suffix must start with '.'.")

    def materialize(self, output_dir: Path | str | None = None) -> Path:
        """
        Render the video if needed and return the resulting file path.

        Parameters
        ----------
        output_dir:
            Optional directory where the rendered video should be created.

        Returns
        -------
        Path
            Path to the rendered video file.

        Notes
        -----
        Repeated calls reuse the same file when it is still present. The lock
        prevents two UI refreshes from rendering the same expensive video at
        the same time.
        """
        with self._lock:
            if self._materialized_path is not None and self._is_valid_video_file(self._materialized_path):
                return self._materialized_path

            artifact_dir = (
                Path(output_dir)
                if output_dir is not None
                else Path(tempfile.gettempdir()) / "sef_video_artifacts"
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            target_path = artifact_dir / f"{self.artifact_id}{self.filename_suffix}"

            rendered = self.render_to(target_path)
            rendered_path = rendered.path if isinstance(rendered, VideoFileArtifact) else Path(rendered)
            if not self._is_valid_video_file(rendered_path):
                raise ValueError(f"Deferred video renderer did not create a valid file: {rendered_path}")

            object.__setattr__(self, "_materialized_path", rendered_path)
            return rendered_path

    @staticmethod
    def _is_valid_video_file(path: Path) -> bool:
        return path.exists() and path.is_file() and path.stat().st_size > 0


VideoLikeArtifact = VideoArtifact | VideoFileArtifact | DeferredVideoArtifact
VIDEO_ARTIFACT_TYPES = (VideoArtifact, VideoFileArtifact, DeferredVideoArtifact)


@dataclass(frozen=True, slots=True, kw_only=True)
class TableArtifact(VisualArtifact):
    """
    Tabular artifact represented as column names and row mappings.

    Rows are copied to dictionaries during construction so adapters can render a
    stable snapshot.
    """

    columns: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "rows", tuple(dict(row) for row in self.rows))


@dataclass(frozen=True, slots=True, kw_only=True)
class JsonArtifact(VisualArtifact):
    """Structured artifact for JSON-like payloads."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "payload", dict(self.payload))


@dataclass(frozen=True, slots=True, kw_only=True)
class TextArtifact(VisualArtifact):
    """
    Textual artifact with an explicit content type.

    Markdown is the default content type because it renders well in docs,
    notebooks, and Streamlit-like adapters.
    """

    content: str
    content_type: str = "text/markdown"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.content_type:
            raise ValueError("TextArtifact.content_type must be a non-empty string.")
