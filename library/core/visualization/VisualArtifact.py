from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualArtifact(ABC):
    """Base contract for UI-agnostic presentation artifacts."""

    artifact_id: str = field(default_factory=lambda: uuid4().hex)
    kind: str
    title: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("VisualArtifact.kind must be a non-empty string.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageArtifact(VisualArtifact):
    """Binary image artifact ready for rendering or persistence."""

    mime_type: str
    data: bytes

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.mime_type:
            raise ValueError("ImageArtifact.mime_type must be a non-empty string.")
        if not self.data:
            raise ValueError("ImageArtifact.data cannot be empty.")


@dataclass(frozen=True, slots=True, kw_only=True)
class TableArtifact(VisualArtifact):
    """Tabular artifact represented as simple records."""

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
    """Textual artifact with an explicit content type."""

    content: str
    content_type: str = "text/markdown"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.content_type:
            raise ValueError("TextArtifact.content_type must be a non-empty string.")
