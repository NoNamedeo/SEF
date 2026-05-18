from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from library.core.artifacts.IntermediateFrameArtifacts import IntermediateFrameArtifactCollection
from library.core.visualization.VisualArtifact import VisualArtifact


@dataclass(frozen=True)
class PipelineExecutionResult:
    """
    Raw execution result before public metadata and reproducibility exports.

    Executors return this DTO to keep execution independent from presentation
    details. ``PipelineOutputAssembler`` is responsible for translating it into
    the stable public ``PipelineOutputs`` contract.
    """

    results: tuple[Any, ...]
    final_artifacts: tuple[VisualArtifact, ...]
    debug_artifacts: tuple[VisualArtifact, ...]
    intermediate_frames: IntermediateFrameArtifactCollection
    latency_policy_metrics: Mapping[str, Any]
