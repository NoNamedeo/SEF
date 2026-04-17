from __future__ import annotations

import logging
from typing import Any, Mapping
from uuid import uuid4

from library.core.events.Event import Event
from library.core.events.PipelineEvent import PipelineEvent
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.interfaces.pipeline.IPipelineFactory import IPipelineFactory
from library.core.interfaces.pipeline.IPipelineRunner import IPipelineRunner
from library.core.pipeline.DefaultPipelineFactory import DefaultPipelineFactory
from library.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor
from library.core.pipeline.Pipeline import Pipeline
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineErrors import (
    InvalidPipelineTriggerEventError,
    PipelineRunAlreadyActiveError,
)
from library.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner
from library.core.visualization.PipelineOutputs import PipelineOutputs

__all__ = ["PipelineOrchestrator"]
log = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Application-facing pipeline execution facade.

    The orchestrator is the single public access point for execution:
    synchronous runs go through ``run(context)``, background runs go through
    ``submit(context)``, and event-driven integrations are optional adapters
    around the same submit path.
    """

    def __init__(
        self,
        runner: IPipelineRunner | None = None,
        pipeline_factory: IPipelineFactory | None = None,
        bus: IEventBus | None = None,
        domain_bus: IEventBus | None = None,
    ) -> None:
        self._runner = runner or ThreadedPipelineRunner(
            monitor=InMemoryPipelineMonitor(),
            lifecycle_bus=bus,
        )
        self._pipeline_factory = pipeline_factory or DefaultPipelineFactory()
        self._bus = bus
        self._domain_bus = domain_bus

        if self._bus is not None:
            self._bus.subscribe(PipelineEvent.event_type, self._on_pipeline_event)

    def run(
        self,
        context: PipelineContext,
        pipeline_id: str | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> PipelineOutputs:
        """
        Execute a pipeline synchronously and return analyzer results.

        This path does not require an EventBus. If a domain bus was configured,
        it is injected into event-emitting components before execution.
        """
        resolved_pipeline_id = pipeline_id or self._new_pipeline_id()
        pipeline = self._build_pipeline(context, resolved_pipeline_id, execution_metadata=execution_metadata)

        return self._runner.run(resolved_pipeline_id, pipeline)

    def submit(
        self,
        context: PipelineContext,
        pipeline_id: str | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Submit a pipeline for background execution and return its id."""
        resolved_pipeline_id = pipeline_id or self._new_pipeline_id()
        pipeline = self._build_pipeline(context, resolved_pipeline_id, execution_metadata=execution_metadata)

        self._runner.submit(resolved_pipeline_id, pipeline)
        return resolved_pipeline_id

    def terminate(self, pipeline_id: str) -> bool:
        """
        Best-effort cancellation for a queued async pipeline.

        Returns True only when the underlying runner cancelled work that had
        not started yet. Already-running pipelines are not interrupted.

        :param pipeline_id: the unique identifier of the pipeline to cancel
        :type pipeline_id: str
        """
        return self._runner.cancel(pipeline_id)

    def active_ids(self) -> list[str]:
        """
        Returns a list of currently active pipeline ids.

        This list is a snapshot of the pipelines currently being executed.
        It does not imply that the pipelines are still running at the time of
        calling this method.

        :return: list of active pipeline ids
        :rtype: list[str]
        """
        return self._runner.active_ids()

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the PipelineOrchestrator.

        This method shuts down the underlying pipeline runner and its executor pool.
        If wait is True, this method blocks until all currently running pipelines
        have finished execution. If wait is False, this method does not block and
        returns immediately.

        :param wait: whether to wait for all currently running pipelines to finish
        :type wait: bool
        """
        self._runner.shutdown(wait=wait)

    def _on_pipeline_event(self, event: Event) -> None:
        try:
            trigger = PipelineEvent.parse(event)
        except InvalidPipelineTriggerEventError as exc:
            log.warning("Ignored invalid pipeline trigger event: %s", exc)
            return

        try:
            self.submit(
                trigger.context,
                pipeline_id=trigger.pipeline_id,
                execution_metadata=trigger.execution_metadata,
            )
        except PipelineRunAlreadyActiveError:
            log.info("Pipeline trigger ignored because '%s' is already running.", trigger.pipeline_id)
        except Exception:
            log.exception("Pipeline trigger submit failed for %s", trigger.pipeline_id)

    def _build_pipeline(
        self,
        context: PipelineContext,
        pipeline_id: str,
        *,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> Pipeline:
        return self._pipeline_factory.create(
            context,
            event_bus=self._domain_bus,
            pipeline_id=pipeline_id,
            execution_metadata=execution_metadata,
        )

    @staticmethod
    def _new_pipeline_id() -> str:
        return f"pipeline-{uuid4().hex[:12]}"
