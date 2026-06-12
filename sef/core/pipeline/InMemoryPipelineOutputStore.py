from __future__ import annotations

import threading
from collections import OrderedDict

from library.core.interfaces.pipeline.IPipelineOutputStore import IPipelineOutputStore
from library.core.visualization.PipelineOutputs import PipelineOutputs


class InMemoryPipelineOutputStore(IPipelineOutputStore):
    """Thread-safe bounded in-memory output store."""

    def __init__(self, max_entries: int = 32) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than 0.")
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._outputs: OrderedDict[str, PipelineOutputs] = OrderedDict()

    def save(self, pipeline_id: str, outputs: PipelineOutputs) -> None:
        with self._lock:
            self._outputs[pipeline_id] = outputs
            self._outputs.move_to_end(pipeline_id)
            while len(self._outputs) > self._max_entries:
                self._outputs.popitem(last=False)

    def get(self, pipeline_id: str) -> PipelineOutputs | None:
        with self._lock:
            return self._outputs.get(pipeline_id)

    def delete(self, pipeline_id: str) -> None:
        with self._lock:
            self._outputs.pop(pipeline_id, None)
