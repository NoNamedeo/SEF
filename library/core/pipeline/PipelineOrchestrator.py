from __future__ import annotations

import logging

# from library.core.artifacts.PipelineEvent import PipelineEvent
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.interfaces.pipeline.IPipelineBuilder import IPipelineBuilder
from library.core.interfaces.pipeline.IPipelineMonitor import IPipelineMonitor
from library.core.interfaces.pipeline.IPipelineRunner import IPipelineRunner

__all__ = ["PipelineOrchestrator"]
log = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Pure facade — zero business logic, zero concrete dependencies.

    Wiring
    ------
    On construction the orchestrator subscribes to *bus* for
    ``PipelineEvent`` triggers.  Each incoming event is forwarded:

        builder.build(event)  →  monitor.register(id)  →  runner.submit(id, pipeline)

    Public interface
    ----------------
    Only two methods are exposed:
    * ``terminate(pipeline_id)`` — cancels execution and removes from tracking.
    * ``active_ids()``           — returns the set of currently running ids.
    """

    def __init__(
        self,
        builder: IPipelineBuilder,
        runner: IPipelineRunner,
        monitor: IPipelineMonitor,
        bus: IEventBus,
    ) -> None:
        self._builder = builder
        self._runner = runner
        self._monitor = monitor
        self._bus = bus
        # self._bus.subscribe(PipelineEvent.event_type, self._on_pipeline_event)

    def terminate(self, pipeline_id: str) -> None:
        """
        Cancels the execution of the pipeline with the given id and removes it from tracking.

        Idempotent: if the pipeline is not running, this method does nothing.

        :param pipeline_id: the unique identifier of the pipeline to cancel
        :type pipeline_id: str
        """
        self._runner.cancel(pipeline_id)
        self._monitor.terminate(pipeline_id)

    def active_ids(self) -> list[str]:
        """
        Returns a list of currently active pipeline ids.

        This list is a snapshot of the pipelines currently being executed.
        It does not imply that the pipelines are still running at the time of
        calling this method.

        :return: list of active pipeline ids
        :rtype: list[str]
        """
        return self._monitor.active_ids()

    def _on_pipeline_event(self, event) -> None:
        try:
            pipeline = self._builder.build(event)
        except Exception as exc:
            log.error("Build failed for %s: %s", event.pipeline_id, exc)
            return

        try:
            self._monitor.register(event.pipeline_id)
            self._runner.submit(event.pipeline_id, pipeline)
        except Exception as exc:
            log.error("Submit failed for %s: %s", event.pipeline_id, exc)
            self._monitor.terminate(event.pipeline_id)  # Rollback
