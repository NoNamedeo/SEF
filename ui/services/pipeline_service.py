"""
Application service layer for the Streamlit UI.

The UI talks to the pipeline core only through this module.  This keeps pages
focused on interaction and presentation while this service owns orchestration,
runner lifecycle, monitor snapshots and event recording.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from library.core.events.Event import Event  # noqa: E402
from library.core.events.EventBus import EventBus  # noqa: E402
from library.core.events.PipelineEvent import PipelineEvent  # noqa: E402
from library.core.pipeline.BranchingCoordinator import BranchingCoordinator  # noqa: E402
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder  # noqa: E402
from library.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor  # noqa: E402
from library.core.pipeline.InMemoryPipelineOutputStore import InMemoryPipelineOutputStore  # noqa: E402
from library.core.pipeline.PipelineContext import PipelineContext  # noqa: E402
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator  # noqa: E402
from library.core.pipeline.PipelineRunSnapshot import PipelineRunSnapshot  # noqa: E402
from library.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner  # noqa: E402
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry  # noqa: E402
from library.core.visualization.PipelineOutputs import PipelineOutputs  # noqa: E402
from library.retry_policies.NoRetryPolicy import NoRetryPolicy  # noqa: E402

log = logging.getLogger(__name__)

_monitor: InMemoryPipelineMonitor | None = None
_output_store: InMemoryPipelineOutputStore | None = None
_runner: ThreadedPipelineRunner | None = None
_orchestrator: PipelineOrchestrator | None = None
_runner_max_workers = 2
_lifecycle_bus: EventBus | None = None
_domain_bus: EventBus | None = None
_branching_coordinator: BranchingCoordinator | None = None
_branching_rule_names: tuple[str, ...] = ()
_event_lock = threading.Lock()
_event_records: list[Event] = []


def _record_event(event: Event) -> None:
    """Store a bounded in-memory event log for UI observability."""
    with _event_lock:
        _event_records.append(event)
        del _event_records[:-250]


def _get_lifecycle_bus() -> EventBus:
    global _lifecycle_bus
    if _lifecycle_bus is None:
        _lifecycle_bus = EventBus()
        _lifecycle_bus.subscribe(EventBus.WILDCARD, _record_event)
    return _lifecycle_bus


def _get_domain_bus() -> EventBus:
    global _domain_bus
    if _domain_bus is None:
        _domain_bus = EventBus()
        _domain_bus.subscribe(EventBus.WILDCARD, _record_event)
    return _domain_bus


def _get_monitor() -> InMemoryPipelineMonitor:
    global _monitor
    if _monitor is None:
        _monitor = InMemoryPipelineMonitor()
    return _monitor


def _get_output_store() -> InMemoryPipelineOutputStore:
    global _output_store
    if _output_store is None:
        _output_store = InMemoryPipelineOutputStore(max_entries=4)
    return _output_store


def _get_runner() -> ThreadedPipelineRunner:
    global _runner
    if _runner is None:
        _runner = ThreadedPipelineRunner(
            monitor=_get_monitor(),
            output_store=_get_output_store(),
            retry_policy=NoRetryPolicy(),
            lifecycle_bus=_get_lifecycle_bus(),
            max_workers=_runner_max_workers,
        )
    return _runner


def _get_orchestrator() -> PipelineOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator(
            runner=_get_runner(),
            bus=_get_lifecycle_bus(),
            domain_bus=_get_domain_bus(),
        )
    return _orchestrator


def run_sync(context: PipelineContext, pipeline_id: str | None = None) -> PipelineOutputs:
    """Execute a pipeline synchronously and return full pipeline outputs."""
    return _get_orchestrator().run(context, pipeline_id=pipeline_id)


def submit_async(pipeline_id: str, context: PipelineContext) -> str:
    """Submit a pipeline for background execution and return its id."""
    submitted_id = _get_orchestrator().submit(context, pipeline_id=pipeline_id)
    log.info("Pipeline '%s' submitted.", submitted_id)
    return submitted_id


def cancel_async(pipeline_id: str) -> bool:
    """Cancel a queued background pipeline when possible."""
    return _get_orchestrator().terminate(pipeline_id)


def active_ids() -> list[str]:
    """Return IDs of currently active pipelines."""
    return _get_orchestrator().active_ids()


def configure_runner(max_workers: int) -> tuple[bool, str]:
    """
    Configure background runner parallelism for future runs.

    Existing active pipelines keep their current executor. When no run is
    active, changing this value safely swaps the UI runner without touching the
    core pipeline implementation.
    """
    global _runner, _orchestrator, _runner_max_workers
    parsed = int(max_workers)
    if parsed <= 0:
        return False, "max_workers deve essere maggiore di zero."
    if parsed == _runner_max_workers:
        return True, f"Runner configurato con {parsed} worker."
    if _runner is not None and _runner.active_ids():
        return False, "Ci sono pipeline attive: cambia i worker quando il runner e idle."

    if _runner is not None:
        _runner.shutdown(wait=False)
    _runner = None
    _orchestrator = None
    _runner_max_workers = parsed
    return True, f"Runner aggiornato a {parsed} worker."


def runner_parallelism() -> int:
    """Return the configured background worker count."""
    return _runner_max_workers


def snapshots() -> list[PipelineRunSnapshot]:
    """Return rich snapshots for all known pipeline runs."""
    return _get_runner().snapshots()


def pipeline_outputs(pipeline_id: str) -> PipelineOutputs | None:
    """Return persisted outputs for a completed pipeline, if available."""
    return _get_output_store().get(pipeline_id)


def event_records() -> list[Event]:
    """Return a snapshot of recorded lifecycle and domain events."""
    with _event_lock:
        return list(_event_records)


def clear_event_records() -> None:
    """Clear the UI event log."""
    with _event_lock:
        _event_records.clear()


def dispatch_trigger(pipeline_id: str, context: PipelineContext) -> None:
    """Start a pipeline through the trigger event path instead of direct submit."""
    _get_lifecycle_bus().dispatch(
        PipelineEvent.create(
            pipeline_id=pipeline_id,
            context=context,
            source="SEFStudio",
        )
    )


def configure_branching_rules(registry: PluginRegistry, rule_names: list[str]) -> tuple[bool, str]:
    """
    Attach BranchingCoordinator once for the selected branching rules.

    The current BranchingCoordinator contract does not support live unsubscription
    or hot-swapping rules, so changing an already configured set requires a server
    restart. Initial activation is fully supported.
    """
    global _branching_coordinator, _branching_rule_names

    selected = tuple(dict.fromkeys(name for name in rule_names if name))
    if selected == _branching_rule_names:
        return True, "Branching già configurato."
    if _branching_coordinator is not None and selected != _branching_rule_names:
        return (
            False,
            "Il core attuale non supporta hot-swap delle branching rules: riavvia Streamlit per cambiare set di regole.",
        )
    if not selected:
        _branching_rule_names = ()
        return True, "Nessuna branching rule attiva."

    rules = [registry.create(PluginCategory.BRANCHING_RULE, name) for name in selected]
    _branching_coordinator = BranchingCoordinator(
        event_bus=_get_domain_bus(),
        rules=rules,
        trigger_bus=_get_lifecycle_bus(),
    )
    _branching_rule_names = selected
    return True, f"Branching attivato con {len(selected)} regola/e."


def event_integration_status() -> dict[str, object]:
    """Return current UI-facing status for lifecycle/domain/branching integration."""
    return {
        "lifecycle_bus": _lifecycle_bus is not None,
        "domain_bus": _domain_bus is not None,
        "branching_enabled": _branching_coordinator is not None,
        "branching_rules": list(_branching_rule_names),
    }


def context_from_config(config: dict[str, Any], registry: PluginRegistry) -> PipelineContext:
    """Build a PipelineContext from a config dictionary using ConfigPipelineBuilder."""
    return ConfigPipelineBuilder(registry).build_context(_normalise_config(config))


def _normalise_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Accept both the current ConfigPipelineBuilder schema and the older UI shape.

    New UI-generated configs already pass constructor-specific params.  Older
    saved configs may still put frame-extractor options directly in params; this
    moves them under OpenCVBufferedFrameExtractor.config.
    """
    pipeline = dict(config.get("pipeline", config))
    normalised = {"pipeline": pipeline}

    frame_extractor = pipeline.get("frame_extractor")
    if isinstance(frame_extractor, dict):
        params = dict(frame_extractor.get("params", {}))
        extractor_config = dict(params.get("config", {}))
        for key in ("resize", "stride", "max_frames"):
            if key in params:
                extractor_config[key] = params.pop(key)
        if extractor_config:
            params["config"] = extractor_config
        frame_extractor["params"] = params

    frame_source_path = None
    if isinstance(frame_extractor, dict):
        params = frame_extractor.get("params", {})
        if isinstance(params, dict):
            raw_path = params.get("path")
            if isinstance(raw_path, str) and raw_path:
                frame_source_path = raw_path

    signal_extractor = pipeline.get("signal_extractor")
    if isinstance(signal_extractor, dict):
        params = dict(signal_extractor.get("params", {}))
        extractor_config = dict(params.get("config", {}))
        extractor_name = str(signal_extractor.get("name", ""))
        if (
            extractor_name in {"opencv_tracker", "opencv_stream_tracker", "opencv_multi_tracker"}
            and frame_source_path
            and "source_path" not in extractor_config
        ):
            extractor_config["source_path"] = frame_source_path
        if extractor_config:
            params["config"] = extractor_config
        signal_extractor["params"] = params

    return normalised
