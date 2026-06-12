from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from library.core.events.PipelineLifecycleEvent import (
    PipelineLifecycleEvent,
    create_pipeline_lifecycle_event,
)
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.interfaces.pipeline.IPipelineMonitor import IPipelineMonitor
from library.core.interfaces.pipeline.IPipelineOutputStore import IPipelineOutputStore
from library.core.interfaces.pipeline.IPipelineRunner import IPipelineRunner
from library.core.interfaces.pipeline.IRetryPolicy import IRetryPolicy
from library.core.pipeline.Pipeline import Pipeline, PipelineExecutionError
from library.core.pipeline.PipelineErrors import PipelineRunAlreadyActiveError
from library.core.pipeline.PipelineRunSnapshot import PipelineRunSnapshot, PipelineRunState
from library.core.visualization.PipelineOutputs import PipelineOutputs
from library.retry_policies.NoRetryPolicy import NoRetryPolicy

log = logging.getLogger(__name__)


class ThreadedPipelineRunner(IPipelineRunner):
    """
    Execute pipelines on a `ThreadPoolExecutor`.

    `run()` and `submit()` share active-id tracking: the same pipeline id cannot
    be executed concurrently through either entry point. `submit()` returns the
    underlying `Future` so callers can observe asynchronous results or failures.

    `cancel()` is best-effort: it can cancel queued work that has not started,
    but it cannot interrupt a pipeline that is already running.

    Lifecycle events are dispatched synchronously to the configured `IEventBus`
    from within the worker thread.

    Thread safety
    -------------
    Runner bookkeeping is protected by a lock. Component instances inside a
    submitted `Pipeline` remain responsible for their own thread-safety.
    """

    def __init__(
        self,
        monitor: IPipelineMonitor,
        output_store: IPipelineOutputStore | None = None,
        retry_policy: IRetryPolicy | None = None,
        lifecycle_bus: IEventBus | None = None,
        max_workers: int = 4,
    ) -> None:
        self._monitor = monitor
        self._output_store = output_store
        self._retry_policy = retry_policy or NoRetryPolicy()
        self._lifecycle_bus = lifecycle_bus
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sef-pipeline",
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future[PipelineOutputs]] = {}
        self._reserved_ids: set[str] = set()

    def run(self, pipeline_id: str, pipeline: Pipeline) -> PipelineOutputs:
        """
        Execute a pipeline synchronously through the retry/lifecycle path.

        Raises
        ------
        PipelineRunAlreadyActiveError
            If another run with the same id is active or queued.
        PipelineExecutionError
            If execution fails and retry policy declines another attempt.
        """
        self._begin_sync(pipeline_id)
        return self._execute(pipeline_id, pipeline)

    def submit(self, pipeline_id: str, pipeline: Pipeline) -> Future[PipelineOutputs]:
        """
        Submit a pipeline for background execution.

        Parameters
        ----------
        pipeline_id:
            Stable run identifier used by monitor, output store, and lifecycle
            events.
        pipeline:
            Already-built pipeline facade.

        Returns
        -------
        Future[PipelineOutputs]
            Future that resolves to completed pipeline outputs or raises the
            execution failure.

        Raises
        ------
        PipelineRunAlreadyActiveError
            If another run with the same id is active or queued.
        Exception
            Any monitor or executor failure raised while reserving the run id.
        """
        rejected_error: PipelineRunAlreadyActiveError | None = None
        submit_failed_error: Exception | None = None
        future: Future[PipelineOutputs] | None = None
        with self._lock:
            if pipeline_id in self._reserved_ids:
                rejected_error = PipelineRunAlreadyActiveError(f"Pipeline '{pipeline_id}' is already running.")
            else:
                self._reserved_ids.add(pipeline_id)
                registered = False
                try:
                    self._clear_previous_outputs(pipeline_id)
                    self._monitor.register(pipeline_id)
                    registered = True
                    future = self._executor.submit(
                        self._execute,
                        pipeline_id,
                        pipeline,
                    )
                except Exception as exc:
                    self._reserved_ids.discard(pipeline_id)
                    if registered:
                        self._monitor.fail(pipeline_id, exc, attempt=0)
                    submit_failed_error = exc
                else:
                    self._futures[pipeline_id] = future

        if rejected_error is not None:
            self._emit(PipelineLifecycleEvent.REJECTED, pipeline_id, error=rejected_error, attempt=0)
            raise rejected_error
        if submit_failed_error is not None:
            self._emit(PipelineLifecycleEvent.SUBMIT_FAILED, pipeline_id, error=submit_failed_error, attempt=0)
            raise submit_failed_error
        if future is None:
            raise RuntimeError(f"Pipeline '{pipeline_id}' submission did not create a Future.")
        return future

    def cancel(self, pipeline_id: str) -> bool:
        """
        Cancel queued work for a pipeline id when possible.

        Returns
        -------
        bool
            `True` only when the underlying `Future` accepted cancellation.
            Already-running pipelines cannot be interrupted through this method.
        """
        with self._lock:
            future = self._futures.get(pipeline_id)
            if future is None:
                return False
            if not future.cancel():
                return False
            self._futures.pop(pipeline_id, None)
            self._reserved_ids.discard(pipeline_id)
        self._delete_outputs(pipeline_id)
        self._monitor.terminate(pipeline_id)
        self._emit(PipelineLifecycleEvent.CANCELLED, pipeline_id, attempt=0)
        return True

    def active_ids(self) -> list[str]:
        """Return the monitor-backed snapshot of active pipeline ids."""
        return self._monitor.active_ids()

    def snapshot(self, pipeline_id: str) -> PipelineRunSnapshot | None:
        """Return the latest known state for a pipeline id."""
        return self._monitor.snapshot(pipeline_id)

    def snapshots(self) -> list[PipelineRunSnapshot]:
        """Return latest known state for all tracked pipeline ids."""
        return self._monitor.snapshots()

    def shutdown(self, wait: bool = True) -> None:
        """
        Shut down the executor.

        With ``wait=False`` pending tasks are cancelled best-effort. Running
        pipelines are not interrupted and clean themselves up through _finish.
        """
        if not wait:
            for pipeline_id in self._cancel_pending_futures():
                self._monitor.terminate(pipeline_id)
                self._emit(PipelineLifecycleEvent.CANCELLED, pipeline_id, attempt=0)
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _execute(self, pipeline_id: str, pipeline: Pipeline) -> PipelineOutputs:
        attempt = 1
        try:
            self._emit(PipelineLifecycleEvent.BEFORE_RUN, pipeline_id)
            while True:
                self._monitor.mark_running(pipeline_id, attempt)
                try:
                    outputs = pipeline.run()
                    self._save_outputs(pipeline_id, outputs)
                    self._emit(
                        PipelineLifecycleEvent.AFTER_RUN,
                        pipeline_id,
                        result_count=len(outputs.results),
                        artifact_count=outputs.artifact_count,
                    )
                    self._monitor.complete(pipeline_id)
                    return outputs
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
                        self._monitor.fail(pipeline_id, exc, attempt)
                        raise
                    next_attempt = attempt + 1
                    self._emit(
                        PipelineLifecycleEvent.ON_RETRY,
                        pipeline_id,
                        attempt=next_attempt,
                    )
                    delay = self._retry_policy.wait_seconds(attempt)
                    if delay > 0:
                        time.sleep(delay)
                    attempt = next_attempt
                except Exception as exc:
                    self._delete_outputs(pipeline_id)
                    self._monitor.fail(pipeline_id, exc, attempt)
                    raise
        except Exception as exc:
            self._delete_outputs(pipeline_id)
            self._fail_if_not_terminal(pipeline_id, exc, attempt)
            raise
        finally:
            self._finish(pipeline_id)

    def _begin_sync(self, pipeline_id: str) -> None:
        rejected_error: PipelineRunAlreadyActiveError | None = None
        with self._lock:
            if pipeline_id in self._reserved_ids:
                rejected_error = PipelineRunAlreadyActiveError(f"Pipeline '{pipeline_id}' is already running.")
            else:
                self._reserved_ids.add(pipeline_id)
                try:
                    self._clear_previous_outputs(pipeline_id)
                    self._monitor.register(pipeline_id)
                except Exception:
                    self._reserved_ids.discard(pipeline_id)
                    raise
        if rejected_error is not None:
            self._emit(PipelineLifecycleEvent.REJECTED, pipeline_id, error=rejected_error, attempt=0)
            raise rejected_error

    def _finish(self, pipeline_id: str) -> None:
        with self._lock:
            self._futures.pop(pipeline_id, None)
            self._reserved_ids.discard(pipeline_id)

    def _fail_if_not_terminal(
        self,
        pipeline_id: str,
        error: Exception | str,
        attempt: int,
    ) -> None:
        snapshot = self._monitor.snapshot(pipeline_id)
        if snapshot is None or snapshot.state in {PipelineRunState.QUEUED, PipelineRunState.RUNNING}:
            self._monitor.fail(pipeline_id, error, attempt)

    def _cancel_pending_futures(self) -> list[str]:
        cancelled_ids: list[str] = []
        with self._lock:
            for pipeline_id, future in list(self._futures.items()):
                if not future.cancel():
                    continue
                self._futures.pop(pipeline_id, None)
                self._reserved_ids.discard(pipeline_id)
                cancelled_ids.append(pipeline_id)
                self._delete_outputs(pipeline_id)
        return cancelled_ids

    def _clear_previous_outputs(self, pipeline_id: str) -> None:
        if self._output_store is None:
            return
        self._output_store.delete(pipeline_id)

    def _save_outputs(self, pipeline_id: str, outputs: PipelineOutputs) -> None:
        if self._output_store is None:
            return
        self._output_store.save(pipeline_id, outputs)

    def _delete_outputs(self, pipeline_id: str) -> None:
        if self._output_store is None:
            return
        self._output_store.delete(pipeline_id)

    def _emit(
        self,
        event: PipelineLifecycleEvent,
        pipeline_id: str,
        result_count: int | None = None,
        artifact_count: int | None = None,
        error: Exception | None = None,
        attempt: int = 1,
    ) -> None:
        if self._lifecycle_bus is None:
            return
        lifecycle_event = create_pipeline_lifecycle_event(
            event=event,
            pipeline_id=pipeline_id,
            source=type(self).__name__,
            result_count=result_count,
            artifact_count=artifact_count,
            error=error,
            attempt=attempt,
        )
        self._lifecycle_bus.dispatch(lifecycle_event)
