from __future__ import annotations

import threading

from library.core.interfaces.pipeline.IPipelineMonitor import IPipelineMonitor


class InMemoryPipelineMonitor(IPipelineMonitor):
    """Thread-safe in-memory implementation of IPipelineMonitor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: set[str] = set()

    def register(self, pipeline_id: str) -> None:
        with self._lock:
            self._active.add(pipeline_id)

    def complete(self, pipeline_id: str) -> None:
        with self._lock:
            self._active.discard(pipeline_id)

    def terminate(self, pipeline_id: str) -> None:
        with self._lock:
            self._active.discard(pipeline_id)

    def active_ids(self) -> list[str]:
        with self._lock:
            return list(self._active)
