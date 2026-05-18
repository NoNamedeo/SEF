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
    """
    Frame pipeline output plus pending streaming work that still must drain.

    ``FramePipelineExecutor`` returns this object before the whole pipeline is
    complete. The adaptive executor can either schedule the pending tasks as
    part of a fully streaming tail or materialize the current frame stream for
    batch execution.
    """

    frame_buffer: FrameBuffer
    pending_tasks: tuple[ThreadedStageTask, ...]
    frame_buffers: list[FrameBuffer]


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
