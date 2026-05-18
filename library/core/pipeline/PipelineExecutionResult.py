from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.IntermediateFrameArtifacts import IntermediateFrameArtifactCollection
from library.core.visualization.VisualArtifact import VisualArtifact

ThreadedStageTask = Callable[[ThreadPoolExecutor], Future[Any]]


@dataclass(frozen=True)
class FramePipelineResult:
    """Frame pipeline output plus pending streaming work that still must drain."""

    frame_buffer: FrameBuffer
    pending_tasks: tuple[ThreadedStageTask, ...]
    frame_buffers: list[FrameBuffer]


@dataclass(frozen=True)
class PipelineExecutionResult:
    """Raw execution result before metadata and reproducibility exports are attached."""

    results: tuple[Any, ...]
    final_artifacts: tuple[VisualArtifact, ...]
    debug_artifacts: tuple[VisualArtifact, ...]
    intermediate_frames: IntermediateFrameArtifactCollection
    latency_policy_metrics: Mapping[str, Any]
