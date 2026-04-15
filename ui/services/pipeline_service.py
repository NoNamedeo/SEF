"""
Thin service wrappers around the SEF Pipeline infrastructure.

Design intent
-------------
Pages import ONLY from this module — they never construct Pipeline or
PipelineContext directly.  When the library evolves (e.g. the async
PipelineOrchestrator becomes fully wired) only this file changes.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# ── project-root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from library.core.interfaces.IData import IData                                    # noqa: E402
from library.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor  # noqa: E402
from library.core.pipeline.PipelineContext import PipelineContext                  # noqa: E402
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator        # noqa: E402
from library.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner    # noqa: E402
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry    # noqa: E402
from library.retry_policies.NoRetryPolicy import NoRetryPolicy                    # noqa: E402

log = logging.getLogger(__name__)

# ── Shared runner / monitor (process-level singletons) ────────────────────────
_monitor: InMemoryPipelineMonitor | None = None
_runner:  ThreadedPipelineRunner | None  = None
_orchestrator: PipelineOrchestrator | None = None


def _get_orchestrator() -> PipelineOrchestrator:
    global _monitor, _runner, _orchestrator
    if _monitor is None:
        _monitor = InMemoryPipelineMonitor()
    if _runner is None:
        _runner = ThreadedPipelineRunner(
            monitor=_monitor,
            retry_policy=NoRetryPolicy(),
            max_workers=4,
        )
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator(
            runner=_runner,
            monitor=_monitor,
        )
    return _orchestrator


# ── Synchronous execution ─────────────────────────────────────────────────────

def run_sync(context: PipelineContext) -> list[IData]:
    """Execute *context* synchronously and return results. Blocks the caller."""
    return _get_orchestrator().run(context)


# ── Async / threaded execution ────────────────────────────────────────────────

def submit_async(pipeline_id: str, context: PipelineContext) -> None:
    """Submit *context* for background execution under *pipeline_id*."""
    _get_orchestrator().submit(context, pipeline_id=pipeline_id)
    log.info("Pipeline '%s' submitted.", pipeline_id)


def cancel_async(pipeline_id: str) -> None:
    """Cancel a background pipeline by ID (best-effort)."""
    _get_orchestrator().terminate(pipeline_id)


def active_ids() -> list[str]:
    """Return IDs of currently running pipelines."""
    return _get_orchestrator().active_ids()


# ── Config-dict → PipelineContext ─────────────────────────────────────────────

def context_from_config(config: dict[str, Any], registry: PluginRegistry) -> PipelineContext:
    """
    Build a PipelineContext from a config dictionary using the PluginRegistry.

    Config schema (mirrors ConfigPipelineBuilder)
    ----------------------------------------------
    pipeline:
      frame_extractor:
        name: opencv_buffered
        params: {path: "/video.mp4", ...}
      frame_cleaners:
        - name: smoothing
      signal_extractor:
        name: opencv_tracker
        params: {tracker_type: CSRT, start_box: [x, y, w, h]}
      signal_cleaners:
        - name: moving_average
          params: {window_size: 5}
      analyzers:
        - name: vertical_position
      visualizers: []
    """
    cfg = config.get("pipeline", config)   # tolerate both wrapped and flat dicts

    def _make(category: PluginCategory, entry: dict) -> Any:
        name = entry.get("name")
        if not name:
            raise ValueError(f"Missing 'name' in config entry for category '{category}'.")
        params: dict = entry.get("params", {})
        return registry.create(category, name, **params)

    def _make_list(category: PluginCategory, entries: list[dict]) -> list:
        return [_make(category, e) for e in entries]

    return PipelineContext(
        frame_extractor=_make(PluginCategory.FRAME_EXTRACTOR, cfg["frame_extractor"]),
        signal_extractor=_make(PluginCategory.SIGNAL_EXTRACTOR, cfg["signal_extractor"]),
        analyzers=_make_list(PluginCategory.ANALYZER, cfg.get("analyzers", [])),
        frame_cleaners=_make_list(PluginCategory.FRAME_CLEANER, cfg.get("frame_cleaners", [])),
        signal_cleaners=_make_list(PluginCategory.SIGNAL_CLEANER, cfg.get("signal_cleaners", [])),
        visualizers=_make_list(PluginCategory.VISUALIZER, cfg.get("visualizers", [])),
    )
