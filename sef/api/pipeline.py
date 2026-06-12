from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from sef.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from sef.core.pipeline.Pipeline import Pipeline
from sef.core.pipeline.PipelineConfigVersioning import CURRENT_PIPELINE_CONFIG_VERSION
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineExecutionPlan import PipelineExecutionPlan
from sef.core.pipeline.PipelineExecutionPlanner import PipelineExecutionPlanner
from sef.core.plugins import PluginCategory, PluginRegistry
from sef.api.config import normalize_config
from sef.api.registry import clone_registry, default_registry
from sef.api.stage_refs import ComponentRef, StageRegistry, StageSpec


@dataclass(frozen=True, slots=True)
class PipelineFacade:
    """
    User-facing immutable builder that hides context construction and execution.

    The facade accepts existing plugin names, component classes, component
    instances, and plain Python callables. At execution time it emits the same
    versioned config consumed by ``ConfigPipelineBuilder``, so advanced users can
    still drop down to the core API without a second pipeline model.
    """

    pipeline_id: str | None
    _registry: PluginRegistry
    _frame_extractor: StageSpec | None = None
    _frame_processors: tuple[StageSpec, ...] = ()
    _signal_extractor: StageSpec | None = None
    _signal_cleaners: tuple[StageSpec, ...] = ()
    _analyzers: tuple[StageSpec, ...] = ()
    _visualizers: tuple[StageSpec, ...] = ()
    _intermediate_frames: dict[str, Any] | None = None
    _runtime: dict[str, Any] | None = None
    _source_config: Mapping[str, Any] | None = None

    def frames(self, component: ComponentRef, **params: Any) -> PipelineFacade:
        """Set the frame source using a plugin name, class, instance, or callable."""
        return replace(self, _frame_extractor=StageSpec(component=component, params=dict(params)), _source_config=None)

    def signals(self, component: ComponentRef, **params: Any) -> PipelineFacade:
        """Set the signal extraction stage."""
        return replace(self, _signal_extractor=StageSpec(component=component, params=dict(params)), _source_config=None)

    def extract(self, component: ComponentRef, **params: Any) -> PipelineFacade:
        """Alias for ``signals`` when the stage reads better as extraction."""
        return self.signals(component, **params)

    def process(
        self,
        component: ComponentRef,
        *,
        processor_type: str | None = None,
        **params: Any,
    ) -> PipelineFacade:
        """
        Append a frame processor.

        ``processor_type`` is optional. The facade infers ``frame_buffer`` for
        registered buffer processors and ``single_frame`` otherwise.
        """
        spec = StageSpec(component=component, params=dict(params), processor_type=processor_type)
        return replace(self, _frame_processors=(*self._frame_processors, spec), _source_config=None)

    def clean(self, component: ComponentRef, **params: Any) -> PipelineFacade:
        """Append a signal cleaner."""
        spec = StageSpec(component=component, params=dict(params))
        return replace(self, _signal_cleaners=(*self._signal_cleaners, spec), _source_config=None)

    def analyze(self, component: ComponentRef, **params: Any) -> PipelineFacade:
        """Append an analyzer."""
        spec = StageSpec(component=component, params=dict(params))
        return replace(self, _analyzers=(*self._analyzers, spec), _source_config=None)

    def visualize(
        self,
        component: ComponentRef,
        *,
        result_indices: tuple[int, ...] | list[int] | None = None,
        **params: Any,
    ) -> PipelineFacade:
        """Append a visualizer, optionally bound to selected analyzer results."""
        indices = tuple(result_indices) if result_indices is not None else None
        spec = StageSpec(component=component, params=dict(params), result_indices=indices)
        return replace(self, _visualizers=(*self._visualizers, spec), _source_config=None)

    def runtime(
        self,
        *,
        frame_buffer_size: int | None = None,
        signal_buffer_size: int | None = None,
        data_buffer_size: int | None = None,
        latency: str | None = None,
        latency_params: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> PipelineFacade:
        """Configure bounded-buffer runtime settings with concise names."""
        runtime_config: dict[str, Any] = dict(self._runtime or {})
        if frame_buffer_size is not None:
            runtime_config["frame_buffer_size"] = frame_buffer_size
        if signal_buffer_size is not None:
            runtime_config["signal_buffer_size"] = signal_buffer_size
        if data_buffer_size is not None:
            runtime_config["data_buffer_size"] = data_buffer_size
        if latency is not None or latency_params is not None:
            runtime_config["latency_policy"] = {
                "name": latency or "blocking",
                "params": dict(latency_params or {}),
            }
        runtime_config.update(extra)
        return replace(self, _runtime=runtime_config, _source_config=None)

    def intermediate_frames(
        self,
        *,
        enabled: bool = True,
        visualizers: tuple[ComponentRef, ...] | list[ComponentRef] = (),
        **capture_config: Any,
    ) -> PipelineFacade:
        """Enable intermediate frame capture and optional intermediate visualizers."""
        section: dict[str, Any] = {"enabled": enabled, **capture_config}
        if visualizers:
            section["visualizers"] = tuple(StageSpec(component=item) for item in visualizers)
        return replace(self, _intermediate_frames=section, _source_config=None)

    def resize(self, width: int, height: int) -> PipelineFacade:
        """Set ``pipeline.frame_extractor.params.config.resize`` for common video sources."""
        return self._with_frame_source_config("resize", (int(width), int(height)))

    def stride(self, value: int) -> PipelineFacade:
        """Set ``pipeline.frame_extractor.params.config.stride``."""
        return self._with_frame_source_config("stride", int(value))

    def max_frames(self, value: int | None) -> PipelineFacade:
        """Set ``pipeline.frame_extractor.params.config.max_frames``."""
        return self._with_frame_source_config("max_frames", value)

    def to_config(self) -> dict[str, Any]:
        """Return the versioned config that will be passed to the core builder."""
        config, _ = self._compile()
        return config

    def build_context(self) -> PipelineContext:
        """Build the core ``PipelineContext`` without running the pipeline."""
        config, registry = self._compile()
        return ConfigPipelineBuilder(registry).build_context(config)

    def execution_plan(self) -> PipelineExecutionPlan:
        """Return the runtime execution plan for the current facade."""
        return PipelineExecutionPlanner().build(self.build_context())

    def run(
        self,
        *,
        pipeline_id: str | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ):
        """Build and execute the pipeline through the stable core facade."""
        resolved_pipeline_id = pipeline_id or self.pipeline_id
        return Pipeline(
            self.build_context(),
            pipeline_id=resolved_pipeline_id,
            execution_metadata=dict(execution_metadata or {}),
        ).run()

    def _compile(self) -> tuple[dict[str, Any], PluginRegistry]:
        if self._source_config is not None:
            return deepcopy(dict(self._source_config)), clone_registry(self._registry)

        stage_registry = StageRegistry(clone_registry(self._registry))
        pipeline_config: dict[str, Any] = {
            "frame_extractor": stage_registry.entry(
                PluginCategory.FRAME_EXTRACTOR,
                self._required(self._frame_extractor, "frames"),
            ),
            "frame_processors": [
                stage_registry.frame_processor_entry(spec)
                for spec in self._frame_processors
            ],
            "signal_extractor": stage_registry.entry(
                PluginCategory.SIGNAL_EXTRACTOR,
                self._required(self._signal_extractor, "signals"),
            ),
            "signal_cleaners": [
                stage_registry.entry(PluginCategory.SIGNAL_CLEANER, spec)
                for spec in self._signal_cleaners
            ],
            "analyzers": [
                stage_registry.entry(PluginCategory.ANALYZER, spec)
                for spec in self._analyzers
            ],
            "visualizers": [
                stage_registry.entry(PluginCategory.VISUALIZER, spec)
                for spec in self._visualizers
            ],
        }
        if self._runtime:
            pipeline_config["runtime"] = deepcopy(self._runtime)
        if self._intermediate_frames:
            pipeline_config["intermediate_frames"] = self._compile_intermediate_frames(stage_registry)

        return {
            "schema_version": CURRENT_PIPELINE_CONFIG_VERSION,
            "pipeline": pipeline_config,
        }, stage_registry.registry

    def _compile_intermediate_frames(self, stage_registry: StageRegistry) -> dict[str, Any]:
        section = dict(self._intermediate_frames or {})
        visualizers = section.get("visualizers", ())
        if visualizers:
            section["visualizers"] = [
                stage_registry.entry(PluginCategory.VISUALIZER, spec)
                for spec in visualizers
            ]
        return section

    def _with_frame_source_config(self, key: str, value: Any) -> PipelineFacade:
        frame_extractor = self._required(self._frame_extractor, "frames")
        params = dict(frame_extractor.params)
        config = dict(params.get("config", {}))
        config[key] = value
        params["config"] = config
        return replace(
            self,
            _frame_extractor=StageSpec(component=frame_extractor.component, params=params),
            _source_config=None,
        )

    @staticmethod
    def _required(spec: StageSpec | None, method_name: str) -> StageSpec:
        if spec is None:
            raise ValueError(f"Call .{method_name}(...) before building or running the pipeline.")
        return spec


def pipeline(
    pipeline_id: str | None = None,
    *,
    registry: PluginRegistry | None = None,
    include_builtins: bool = False,
) -> PipelineFacade:
    """
    Create a high-level SEF pipeline facade.

    The generic facade is lightweight by default. Use ``sef.video`` or
    ``sef.webcam`` for built-in OpenCV sources, or pass
    ``include_builtins=True`` when you want to reference built-in plugin names
    directly from ``pipeline``.
    """
    return PipelineFacade(
        pipeline_id=pipeline_id,
        _registry=clone_registry(registry) if registry is not None else default_registry(include_builtins=include_builtins),
    )


def video(
    path: str,
    *,
    pipeline_id: str | None = None,
    registry: PluginRegistry | None = None,
    include_builtins: bool = True,
    **config: Any,
) -> PipelineFacade:
    """Create a facade using the built-in OpenCV buffered video extractor."""
    params: dict[str, Any] = {"path": path}
    if config:
        params["config"] = dict(config)
    return pipeline(pipeline_id, registry=registry, include_builtins=include_builtins).frames("opencv_buffered", **params)


def webcam(
    index: int = 0,
    *,
    pipeline_id: str | None = None,
    registry: PluginRegistry | None = None,
    include_builtins: bool = True,
    **config: Any,
) -> PipelineFacade:
    """Create a facade using the built-in OpenCV webcam extractor."""
    params: dict[str, Any] = {"camera_index": int(index)}
    if config:
        params["config"] = dict(config)
    return pipeline(pipeline_id, registry=registry, include_builtins=include_builtins).frames("opencv_webcam", **params)


def from_config(
    config: Mapping[str, Any],
    *,
    pipeline_id: str | None = None,
    registry: PluginRegistry | None = None,
    include_builtins: bool = True,
) -> PipelineFacade:
    """Create a runnable facade from an existing SEF config mapping."""
    return PipelineFacade(
        pipeline_id=pipeline_id,
        _registry=clone_registry(registry) if registry is not None else default_registry(include_builtins=include_builtins),
        _source_config=normalize_config(config),
    )
