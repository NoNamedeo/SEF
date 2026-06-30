from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from sef.api.registry import clone_registry, default_registry
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineRunOptions import RUN_CONFIG_KEY
from sef.core.plugins import PluginRegistry
from sef.core.visualization import PipelineOutputs


@dataclass(frozen=True, slots=True)
class PreparedRunConfig:
    """Run config plus the local registry needed to materialize it."""

    config: dict[str, Any]
    registry: PluginRegistry


def run(
    pipeline: Any,
    *,
    id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    run: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    include_builtins: bool = True,
) -> PipelineOutputs:
    """
    Execute a pipeline facade or run config through the shared orchestrator path.

    ``PipelineContext`` is intentionally not accepted here; context execution is
    an advanced core concern exposed by ``PipelineOrchestrator.run_context``.
    """
    prepared = prepare_run_config(
        pipeline,
        id=id,
        metadata=metadata,
        run=run,
        registry=registry,
        include_builtins=include_builtins,
    )
    from sef.api.orchestrator import orchestrator

    return orchestrator(registry=prepared.registry, include_builtins=False).run(prepared.config)


def submit(
    pipeline: Any,
    *,
    id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    run: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    include_builtins: bool = True,
) -> Future[PipelineOutputs]:
    """Submit a pipeline facade or run config through the shared orchestrator path."""
    prepared = prepare_run_config(
        pipeline,
        id=id,
        metadata=metadata,
        run=run,
        registry=registry,
        include_builtins=include_builtins,
    )
    from sef.api.orchestrator import orchestrator

    owner = orchestrator(registry=prepared.registry, include_builtins=False)
    future = owner.submit(prepared.config)
    setattr(future, "_sef_orchestrator", owner)
    return future


def prepare_run_config(
    pipeline: Any,
    *,
    id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    run: Mapping[str, Any] | None = None,
    registry: PluginRegistry | None = None,
    include_builtins: bool = True,
) -> PreparedRunConfig:
    """
    Normalize public execution input into a run config and local registry.

    This function is shared by top-level helpers and ``OrchestratorFacade`` so
    every public path reaches the same core orchestrator contract.
    """
    if isinstance(pipeline, PipelineContext):
        raise TypeError(
            "sef.run/submit do not accept PipelineContext. "
            "Use sef.core.PipelineOrchestrator.run_context for advanced context execution."
        )

    if _is_pipeline_facade(pipeline):
        config, scoped_registry = pipeline._compile()  # noqa: SLF001 - package-level facade adapter.
        return PreparedRunConfig(
            config=_apply_run_overrides(
                config,
                id=id,
                metadata=metadata,
                run=run,
            ),
            registry=scoped_registry,
        )

    if isinstance(pipeline, Mapping):
        resolved_registry = clone_registry(registry) if registry is not None else default_registry(include_builtins=include_builtins)
        return PreparedRunConfig(
            config=_apply_run_overrides(
                dict(pipeline),
                id=id,
                metadata=metadata,
                run=run,
            ),
            registry=resolved_registry,
        )

    raise TypeError("run/submit expects a PipelineFacade or config mapping.")


def _is_pipeline_facade(value: Any) -> bool:
    from sef.api.pipeline import PipelineFacade

    return isinstance(value, PipelineFacade)


def _apply_run_overrides(
    config: Mapping[str, Any],
    *,
    id: str | None,
    metadata: Mapping[str, Any] | None,
    run: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(config)
    if id is not None:
        result["id"] = id
    if metadata is not None:
        current_metadata = result.get("metadata", {})
        if not isinstance(current_metadata, Mapping):
            current_metadata = {}
        result["metadata"] = {**dict(current_metadata), **dict(metadata)}

    run_section: dict[str, Any] = {}
    current_run = result.get(RUN_CONFIG_KEY)
    if isinstance(current_run, Mapping):
        run_section.update(dict(current_run))
    if run is not None:
        run_section.update(dict(run))
    if run_section:
        result[RUN_CONFIG_KEY] = run_section
    return result


__all__ = ["run", "submit"]
