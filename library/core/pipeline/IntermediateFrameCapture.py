from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from library.core.artifacts.Frame import Frame
from library.core.artifacts.IntermediateFrameArtifacts import (
    IntermediateFrameArtifactCollection,
    IntermediateFrameArtifactExporter,
)
from library.core.artifacts.MaskArtifacts import IntermediateFrameArtifact


@dataclass(frozen=True, slots=True)
class IntermediateFrameCaptureConfig:
    """
    Configuration for bounded intermediate-frame debugging.

    Capture is opt-in. When enabled, the stage samples processed source frames
    and stores at most ``max_stored_frames`` source frame indexes. Each sampled
    frame may contain one artifact per cleaner stage.
    """

    enabled: bool = False
    sampling_interval: int = 1
    max_stored_frames: int = 50
    export_directory: Path | None = None
    lazy_saving: bool = True
    include_original: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sampling_interval <= 0:
            raise ValueError("IntermediateFrameCaptureConfig.sampling_interval must be greater than 0.")
        if self.max_stored_frames < 0:
            raise ValueError("IntermediateFrameCaptureConfig.max_stored_frames cannot be negative.")
        if self.export_directory is not None:
            object.__setattr__(self, "export_directory", Path(self.export_directory))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def disabled(cls) -> IntermediateFrameCaptureConfig:
        """Return the default disabled capture configuration."""
        return cls(enabled=False)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> IntermediateFrameCaptureConfig:
        """Parse public pipeline configuration keys and aliases."""
        if not config:
            return cls.disabled()
        export_directory = cls._first_present(config, "export_directory", "export_dir")
        sampling_interval = cls._first_present(config, "sampling_interval", "frame_sample_interval", "sample_every_n_frames")
        max_stored_frames = cls._first_present(config, "max_stored_frames", "max_frames")
        return cls(
            enabled=bool(config.get("enabled", True)),
            sampling_interval=cls._positive_int(sampling_interval, default=1, field_name="sampling_interval"),
            max_stored_frames=cls._non_negative_int(max_stored_frames, default=50, field_name="max_stored_frames"),
            export_directory=Path(str(export_directory)) if export_directory is not None else None,
            lazy_saving=bool(config.get("lazy_saving", config.get("deferred_saving", True))),
            include_original=bool(config.get("include_original", True)),
            metadata=dict(config.get("metadata", {}) or {}),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-friendly capture configuration metadata."""
        metadata = dict(self.metadata)
        metadata.update(
            {
                "enabled": self.enabled,
                "sampling_interval": self.sampling_interval,
                "max_stored_frames": self.max_stored_frames,
                "lazy_saving": self.lazy_saving,
                "include_original": self.include_original,
            }
        )
        if self.export_directory is not None:
            metadata["export_directory"] = str(self.export_directory)
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
    def _non_negative_int(value: Any, *, default: int, field_name: str) -> int:
        if value is None:
            return default
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{field_name} cannot be negative.")
        return parsed


@dataclass(frozen=True, slots=True)
class IntermediateFrameCaptureContext:
    """Context passed to optional cleaner-specific intermediate artifact emitters."""

    source_sequence_index: int
    frame_index: int | None
    cleaner_index: int
    cleaner_name: str
    stage_name: str
    timestamp_seconds: float | None
    cleaner_config: Mapping[str, Any]


class IntermediateFrameEmitter(Protocol):
    """
    Optional protocol for frame cleaners that emit custom debug artifacts.

    A cleaner may implement this method to attach masks, overlays, or custom
    metadata. Cleaners that do not implement it still get default before/after
    snapshots from ``FrameCleaningStage`` when capture is enabled.
    """

    def emit_intermediate_artifacts(
        self,
        original_frame: Frame,
        cleaned_frame: Frame,
        context: IntermediateFrameCaptureContext,
    ) -> Iterable[IntermediateFrameArtifact]:
        """Return zero or more artifacts for a single cleaner application."""


class IntermediateFrameArtifactStore:
    """Bounded in-memory store with optional eager export for captured snapshots."""

    def __init__(self, config: IntermediateFrameCaptureConfig) -> None:
        self._config = config
        self._artifacts_by_sequence: OrderedDict[int, list[IntermediateFrameArtifact]] = OrderedDict()
        self._eager_exported_paths: list[Path] = []
        self._exporter = (
            IntermediateFrameArtifactExporter(config.export_directory)
            if config.export_directory is not None and not config.lazy_saving
            else None
        )

    @property
    def config(self) -> IntermediateFrameCaptureConfig:
        """Return the immutable capture configuration."""
        return self._config

    def should_capture(self, source_sequence_index: int) -> bool:
        """Return whether the processed source frame should be captured."""
        return (
            self._config.enabled
            and self._config.max_stored_frames > 0
            and source_sequence_index % self._config.sampling_interval == 0
        )

    def add(
        self,
        artifact: IntermediateFrameArtifact,
        *,
        source_sequence_index: int,
    ) -> None:
        """Store one artifact, evicting the oldest sampled source frame if needed."""
        if not self.should_capture(source_sequence_index):
            return

        if source_sequence_index not in self._artifacts_by_sequence:
            self._artifacts_by_sequence[source_sequence_index] = []
            self._artifacts_by_sequence.move_to_end(source_sequence_index)
            while len(self._artifacts_by_sequence) > self._config.max_stored_frames:
                self._artifacts_by_sequence.popitem(last=False)

        self._artifacts_by_sequence[source_sequence_index].append(artifact)
        if self._exporter is not None:
            self._eager_exported_paths.extend(self._exporter.export(artifact))

    def to_collection(self) -> IntermediateFrameArtifactCollection:
        """Return all retained artifacts as an immutable collection."""
        artifacts = tuple(
            artifact
            for artifacts_for_frame in self._artifacts_by_sequence.values()
            for artifact in artifacts_for_frame
        )
        metadata = self._config.to_metadata()
        if self._eager_exported_paths:
            metadata["eager_exported_paths"] = [str(path) for path in self._eager_exported_paths]
        return IntermediateFrameArtifactCollection(artifacts=artifacts, metadata=metadata)
