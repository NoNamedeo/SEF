from __future__ import annotations

from collections.abc import Iterable
from threading import Lock
from typing import Any

from library.core.artifacts.buffer.FrameBuffer import FrameBuffer
from library.core.interfaces.BufferContracts import IAbortableBuffer
from library.core.pipeline.PipelineBuffers import PipelineBuffers
from library.core.visualization.VisualArtifact import VisualArtifact


class PipelineExecutionResources:
    """
    Shared mutable resources for one pipeline run.

    Segment executors register every bounded buffer they create so a failure can
    abort the whole concurrent graph. Final artifacts are collected here because
    frame exporters and visualizers can both produce public artifacts.
    """

    def __init__(self) -> None:
        self.frame_buffers: list[FrameBuffer] = []
        self.signal_buffers: list[IAbortableBuffer[Any]] = []
        self.data_buffers: list[IAbortableBuffer[Any]] = []
        self._final_artifacts: list[VisualArtifact] = []
        self._artifact_lock = Lock()

    @property
    def final_artifacts(self) -> tuple[VisualArtifact, ...]:
        """Return the artifacts produced by final exporters and visualizers."""
        return tuple(self._final_artifacts)

    def add_final_artifacts(self, artifacts: Iterable[VisualArtifact]) -> None:
        """Append final artifacts from a stage in a thread-safe way."""
        with self._artifact_lock:
            self._final_artifacts.extend(artifacts)

    def abort_all_buffers(self) -> None:
        """Abort every registered buffer to unblock concurrent producers."""
        PipelineBuffers.abort_all(
            self.frame_buffers,
            self.signal_buffers,
            self.data_buffers,
        )
