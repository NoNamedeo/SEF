from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Any, Mapping
from uuid import uuid4

from sef.core.events.Event import Event
from sef.core.events.PipelineEvent import PipelineEvent
from sef.core.interfaces.pipeline.IEventBus import IEventBus
from sef.core.interfaces.pipeline.IPipelineFactory import IPipelineFactory
from sef.core.interfaces.pipeline.IPipelineRunner import IPipelineRunner
from sef.core.pipeline.DefaultPipelineFactory import DefaultPipelineFactory
from sef.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor
from sef.core.pipeline.Pipeline import Pipeline
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineErrors import (
    InvalidPipelineTriggerEventError,
    PipelineRunAlreadyActiveError,
)
from sef.core.pipeline.PipelineRunMaterializer import PipelineRunMaterializer
from sef.core.pipeline.PipelineRunOptions import PipelineRunOptions
from sef.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner
from sef.core.plugins.PluginRegistry import PluginRegistry
from sef.core.visualization.PipelineOutputs import PipelineOutputs

__all__ = ["PipelineOrchestrator"]
log = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Application-facing pipeline execution facade.

    The orchestrator is the single public access point for execution:
    synchronous runs go through `run(context)`, background runs go through
    `submit(context)`, and event-driven integrations are optional adapters
    around the same submit path.

    Boundary
    --------
    The orchestrator coordinates application ports. It does not validate
    component schemas or execute stage logic directly; those responsibilities
    belong to builders, contexts, runners, and pipeline execution collaborators.
    """

    def __init__(
        self,
        runner: IPipelineRunner | None = None,
        pipeline_factory: IPipelineFactory | None = None,
        registry: PluginRegistry | None = None,
        materializer: PipelineRunMaterializer | None = None,
        bus: IEventBus | None = None,
        domain_bus: IEventBus | None = None,
    ) -> None:
        self._runner = runner or ThreadedPipelineRunner(
            monitor=InMemoryPipelineMonitor(),
            lifecycle_bus=bus,
        )
        self._pipeline_factory = pipeline_factory or DefaultPipelineFactory()
        self._materializer = materializer or PipelineRunMaterializer(registry)
        self._bus = bus
        self._domain_bus = domain_bus

        if self._bus is not None:
            self._bus.subscribe(PipelineEvent.event_type, self._on_pipeline_event)

    def run(
        self,
        config: Mapping[str, Any],
    ) -> PipelineOutputs:
        """
        Execute a declarative run config synchronously and return outputs.

        This path does not require an EventBus. If a domain bus was configured,
        it is injected into event-emitting components before execution.

        Parameters
        ----------
        config:
            Run config containing ``pipeline`` and optional ``id``,
            ``metadata`` and ``run`` sections.

        Returns
        -------
        PipelineOutputs
            Completed outputs for the run.
        """
        materialized = self._materializer.materialize(config)
        return self.run_context(
            materialized.context,
            id=materialized.pipeline_id,
            metadata=materialized.execution_metadata,
            run=materialized.run_options.to_config(),
        )

    def submit(
        self,
        config: Mapping[str, Any],
    ) -> Future[PipelineOutputs]:
        """
        Submit a pipeline for background execution.

        Returns
        -------
        Future[PipelineOutputs]
            Future owned by the configured runner.
        """
        materialized = self._materializer.materialize(config)
        return self.submit_context(
            materialized.context,
            id=materialized.pipeline_id,
            metadata=materialized.execution_metadata,
            run=materialized.run_options.to_config(),
        )

    def run_context(
        self,
        context: PipelineContext,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> PipelineOutputs:
        """
        Execute an already-materialized context.

        This is the advanced/core path for tests and integrations that own
        component instances directly.
        """
        resolved_pipeline_id = id or self._new_pipeline_id()
        pipeline = self._build_pipeline(
            context,
            resolved_pipeline_id,
            execution_metadata=metadata,
            run_options=PipelineRunOptions.from_run_mapping(run),
        )
        return self._runner.run(resolved_pipeline_id, pipeline)

    def submit_context(
        self,
        context: PipelineContext,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> Future[PipelineOutputs]:
        """
        Submit an already-materialized context through the configured runner.

        This mirrors ``run_context`` for advanced asynchronous integrations.
        """
        resolved_pipeline_id = id or self._new_pipeline_id()
        pipeline = self._build_pipeline(
            context,
            resolved_pipeline_id,
            execution_metadata=metadata,
            run_options=PipelineRunOptions.from_run_mapping(run),
        )
        return self._runner.submit(resolved_pipeline_id, pipeline)

    def terminate(self, pipeline_id: str) -> bool:
        """
        Best-effort cancellation for a queued async pipeline.

        Parameters
        ----------
        pipeline_id:
            Run identifier returned or supplied at submission time.

        Returns
        -------
        bool
            `True` only when the underlying runner cancelled queued work that
            had not started yet. Already-running pipelines are not interrupted.
        """
        return self._runner.cancel(pipeline_id)

    def active_ids(self) -> list[str]:
        """
        Return a snapshot of currently active pipeline ids.

        This list is a snapshot of the pipelines currently being executed.
        It does not imply that the pipelines are still running at the time of
        calling this method because async runs may complete immediately after
        the snapshot is read.
        """
        return self._runner.active_ids()

    def shutdown(self, wait: bool = True) -> None:
        """
        Shut down the underlying runner.

        Parameters
        ----------
        wait:
            When `True`, block until running pipelines finish. When `False`,
            ask the runner to cancel pending work and return without waiting for
            already-running pipelines.
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
                trigger.config,
            )
        except PipelineRunAlreadyActiveError:
            log.info("Pipeline trigger ignored because '%s' is already running.", trigger.config.get("id", "-"))
        except Exception:
            log.exception("Pipeline trigger submit failed for %s", trigger.config.get("id", "-"))

    def _build_pipeline(
        self,
        context: PipelineContext,
        pipeline_id: str,
        *,
        execution_metadata: Mapping[str, Any] | None = None,
        run_options: PipelineRunOptions | None = None,
    ) -> Pipeline:
        create_kwargs: dict[str, Any] = {
            "event_bus": self._domain_bus,
            "pipeline_id": pipeline_id,
            "execution_metadata": execution_metadata,
        }
        if run_options is not None:
            create_kwargs["run_options"] = run_options
        return self._pipeline_factory.create(context, **create_kwargs)

    @staticmethod
    def _new_pipeline_id() -> str:
        return f"pipeline-{uuid4().hex[:12]}"
