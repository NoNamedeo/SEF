"""Typed models for UI-oriented execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from library.core.visualization.VisualArtifact import VisualArtifact


@dataclass(frozen=True, slots=True)
class ArtifactOutput:
    """Visual artifact produced by configured visualizers or the pipeline."""

    artifact: VisualArtifact
    source: str


@dataclass(frozen=True, slots=True)
class ReconstructedVideoOutput:
    """Explicit reconstructed video payload shown in a dedicated UI section."""

    artifact_id: str
    title: str
    mime_type: str
    data: bytes
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnalysisResultOutput:
    """Analysis result with UI-specific preview artifacts and summaries."""

    result_id: str
    title: str
    type_name: str
    data: Any
    preview_artifacts: tuple[VisualArtifact, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    detail_rows: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionResultsView:
    """Whole execution payload rendered by the UI."""

    analysis_results: tuple[AnalysisResultOutput, ...]
    visualizer_outputs: tuple[ArtifactOutput, ...]
    reconstructed_videos: tuple[ReconstructedVideoOutput, ...]
    metadata: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
