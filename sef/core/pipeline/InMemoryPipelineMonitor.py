from __future__ import annotations

import threading
import time
from dataclasses import replace

from library.core.interfaces.pipeline.IPipelineMonitor import IPipelineMonitor
from library.core.pipeline.PipelineRunSnapshot import PipelineRunSnapshot, PipelineRunState


class InMemoryPipelineMonitor(IPipelineMonitor):
    """Thread-safe in-memory implementation of IPipelineMonitor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: dict[str, PipelineRunSnapshot] = {}

    def register(self, pipeline_id: str) -> None:
        with self._lock:
            self._snapshots[pipeline_id] = PipelineRunSnapshot(
                pipeline_id=pipeline_id,
                state=PipelineRunState.QUEUED,
                submitted_at=time.time(),
            )

    def mark_running(self, pipeline_id: str, attempt: int) -> None:
        with self._lock:
            current = self._snapshot_or_default(pipeline_id)
            self._snapshots[pipeline_id] = replace(
                current,
                state=PipelineRunState.RUNNING,
                attempt=attempt,
                started_at=current.started_at or time.time(),
            )

    def complete(self, pipeline_id: str) -> None:
        with self._lock:
            current = self._snapshot_or_default(pipeline_id)
            self._snapshots[pipeline_id] = replace(
                current,
                state=PipelineRunState.SUCCEEDED,
                error=None,
                completed_at=time.time(),
            )

    def fail(self, pipeline_id: str, error: Exception | str, attempt: int) -> None:
        with self._lock:
            current = self._snapshot_or_default(pipeline_id)
            self._snapshots[pipeline_id] = replace(
                current,
                state=PipelineRunState.FAILED,
                attempt=attempt,
                error=str(error),
                completed_at=time.time(),
            )

    def terminate(self, pipeline_id: str) -> None:
        with self._lock:
            current = self._snapshot_or_default(pipeline_id)
            self._snapshots[pipeline_id] = replace(
                current,
                state=PipelineRunState.CANCELLED,
                completed_at=time.time(),
            )

    def active_ids(self) -> list[str]:
        with self._lock:
            return [
                pipeline_id
                for pipeline_id, snapshot in self._snapshots.items()
                if snapshot.state in {PipelineRunState.QUEUED, PipelineRunState.RUNNING}
            ]

    def snapshot(self, pipeline_id: str) -> PipelineRunSnapshot | None:
        with self._lock:
            return self._snapshots.get(pipeline_id)

    def snapshots(self) -> list[PipelineRunSnapshot]:
        with self._lock:
            return list(self._snapshots.values())

    @staticmethod
    def _default_snapshot(pipeline_id: str) -> PipelineRunSnapshot:
        return PipelineRunSnapshot(
            pipeline_id=pipeline_id,
            state=PipelineRunState.QUEUED,
            submitted_at=time.time(),
        )

    def _snapshot_or_default(self, pipeline_id: str) -> PipelineRunSnapshot:
        return self._snapshots.get(pipeline_id) or self._default_snapshot(pipeline_id)
