from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor

from library.core.events.PipelineLifecycleBus import (
    PipelineLifecycleEvent,
    PipelineLifecyclePayload,
)
from library.core.interfaces.IData import IData
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.interfaces.pipeline.IPipelineMonitor import IPipelineMonitor
from library.core.interfaces.pipeline.IPipelineRunner import IPipelineRunner
from library.core.interfaces.pipeline.IRetryPolicy import IRetryPolicy
from library.core.pipeline.Pipeline import Pipeline, PipelineExecutionError
from library.retry_policies.NoRetryPolicy import NoRetryPolicy

log = logging.getLogger(__name__)


class ThreadedPipelineRunner(IPipelineRunner):
    """
    Executes pipelines on a ThreadPoolExecutor with retry and lifecycle events.

    Lifecycle events (BEFORE_RUN, AFTER_RUN, ON_ERROR, ON_RETRY) are
    dispatched synchronously to the provided IEventBus from within the
    worker thread.
    """

    def __init__(
        self,
        monitor: IPipelineMonitor,
        retry_policy: IRetryPolicy | None = None,
        lifecycle_bus: IEventBus | None = None,
        max_workers: int = 4,
    ) -> None:
        self._monitor = monitor
        self._retry_policy = retry_policy or NoRetryPolicy()
        self._lifecycle_bus = lifecycle_bus
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sef-pipeline",
        )
        self._futures: dict[str, Future[list[IData]]] = {}

    def submit(self, pipeline_id: str, pipeline: Pipeline) -> None:
        future: Future[list[IData]] = self._executor.submit(self._run, pipeline_id, pipeline)
        self._futures[pipeline_id] = future

    def cancel(self, pipeline_id: str) -> None:
        future = self._futures.pop(pipeline_id, None)
        if future is not None and future.cancel():
            self._monitor.complete(pipeline_id)

    def _run(self, pipeline_id: str, pipeline: Pipeline) -> list[IData]:
        try:
            self._emit(PipelineLifecycleEvent.BEFORE_RUN, pipeline_id)
            attempt = 1
            while True:
                try:
                    results = pipeline.run()
                    self._emit(PipelineLifecycleEvent.AFTER_RUN, pipeline_id, results=results)
                    return results
                except PipelineExecutionError as exc:
                    self._emit(
                        PipelineLifecycleEvent.ON_ERROR,
                        pipeline_id,
                        error=exc,
                        attempt=attempt,
                    )
                    log.error(
                        "Pipeline %s failed on attempt %d: %s",
                        pipeline_id,
                        attempt,
                        exc,
                    )
                    if not self._retry_policy.should_retry(attempt, exc):
                        raise
                    delay = self._retry_policy.wait_seconds(attempt)
                    if delay > 0:
                        time.sleep(delay)
                    attempt += 1
                    self._emit(PipelineLifecycleEvent.ON_RETRY, pipeline_id, attempt=attempt)
        finally:
            self._futures.pop(pipeline_id, None)
            self._monitor.complete(pipeline_id)

    def _emit(
        self,
        event: PipelineLifecycleEvent,
        pipeline_id: str,
        results: list[IData] | None = None,
        error: Exception | None = None,
        attempt: int = 1,
    ) -> None:
        if self._lifecycle_bus is None:
            return
        payload = PipelineLifecyclePayload(
            event=event,
            pipeline_id=pipeline_id,
            results=results or [],
            error=error,
            attempt=attempt,
        )
        self._lifecycle_bus.dispatch(payload)
