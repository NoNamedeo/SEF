from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sef.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from sef.core.pipeline.IntermediateFrameCapture import IntermediateFrameCaptureConfig
from sef.core.pipeline.PipelineConfigVersioning import normalize_pipeline_config
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineErrors import (
    ConfigSchemaError,
    PipelineConfigurationError,
    PluginConstructionError,
    PluginResolutionError,
)
from sef.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from sef.core.pipeline.StreamRuntimeConfig import StreamRuntimeConfig
from sef.core.pipeline.VisualizerBinding import VisualizerBinding
from sef.core.plugins.PluginRegistry import PluginCategory, PluginRegistry


class ConfigPipelineBuilder:
    """
    Build a validated `PipelineContext` from a declarative configuration.

    Design rationale
    ----------------
    `ConfigPipelineBuilder` exists for deployment scenarios where the pipeline
    topology is owned by YAML, JSON, an API payload, or a UI editor rather than
    by Python construction code.

    The builder delegates all component construction to `PluginRegistry`. It
    knows category names and schema structure, but it does not import concrete
    frame extractors, processors, analyzers, or visualizers. This keeps the
    core configuration layer independent from infrastructure adapters.

    Configuration schema
    --------------------
    ```yaml
    schema_version: "1.0"
    pipeline:
      frame_extractor:
        name: opencv_buffered
        params:
          path: /path/to/video.mp4

      frame_processors:
        - name: opencv_gray
          processor_type: single_frame
        - name: motion_magnification
          processor_type: frame_buffer

      signal_extractor:
        name: opencv_tracker
        params:
          roi: [100, 200, 50, 80]

      signal_cleaners:
        - name: moving_average
          params:
            window: 5

      analyzers:
        - name: vertical_position

      visualizers:
        - name: matplotlib
          result_indices: [0]

      intermediate_frames:
        enabled: true
        sampling_interval: 10
        max_stored_frames: 20
        export_directory: artifacts/debug
        lazy_saving: true
        visualizers:
          - name: intermediate_frames_grid

      runtime:
        frame_buffer_size: 8
        signal_buffer_size: 8
        data_buffer_size: 8
        latency_policy:
          name: blocking
          params: {}
    ```

    Versioning
    ----------
    Configs are normalized through `normalize_pipeline_config`. Unversioned
    configs are accepted as the current public schema for compatibility, while
    unsupported explicit versions raise `ConfigVersionError`.

    Mutability
    ----------
    The input mapping is read and copied into a new `PipelineContext`; callers
    should not rely on later mutations to the original mapping.

    Raises
    ------
    PipelineConfigurationError
        If a required section is missing, a plugin entry is malformed, a plugin
        cannot be found, plugin construction fails, or the resulting context
        violates PipelineContext invariants.
    """

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    # ── Public API ───────────────────────────────────────────────────────────

    def build_context(self, config: Mapping[str, Any]) -> PipelineContext:
        """
        Resolve a versioned config into an immutable `PipelineContext`.

        Parameters
        ----------
        config:
            Mapping containing a top-level `pipeline` section and, optionally,
            a `schema_version` string.

        Returns
        -------
        PipelineContext
            Validated context containing constructed component instances,
            runtime settings, visualizer bindings, intermediate-frame capture
            configuration, and a compact reproducibility copy of the source
            config.

        Raises
        ------
        ConfigSchemaError
            If required sections are missing or have the wrong shape.
        ConfigVersionError
            If `schema_version` is explicit and unsupported.
        PluginResolutionError
            If the registry cannot resolve a configured component name.
        PluginConstructionError
            If a plugin factory raises or receives invalid parameters.
        PipelineConfigurationError
            If context validation fails after plugin construction.
        """
        try:
            versioned_config = normalize_pipeline_config(config)
            cfg = dict(versioned_config.pipeline)
            return PipelineContext(
                frame_extractor=self._create(
                    PluginCategory.FRAME_EXTRACTOR,
                    self._required_mapping(cfg, "frame_extractor", "pipeline.frame_extractor"),
                    "pipeline.frame_extractor",
                ),
                signal_extractor=self._create(
                    PluginCategory.SIGNAL_EXTRACTOR,
                    self._required_mapping(cfg, "signal_extractor", "pipeline.signal_extractor"),
                    "pipeline.signal_extractor",
                ),
                frame_processors=self._frame_processors(cfg),
                signal_cleaners=self._build_list(
                    PluginCategory.SIGNAL_CLEANER,
                    self._optional_list(cfg, "signal_cleaners", "pipeline.signal_cleaners"),
                    "pipeline.signal_cleaners",
                ),
                analyzers=self._build_list(
                    PluginCategory.ANALYZER,
                    self._required_list(cfg, "analyzers", "pipeline.analyzers"),
                    "pipeline.analyzers",
                ),
                visualizers=self._build_list(
                    PluginCategory.VISUALIZER,
                    self._unbound_visualizers(cfg),
                    "pipeline.visualizers",
                ),
                visualizer_bindings=self._visualizer_bindings(cfg),
                intermediate_frame_capture=self._intermediate_frame_capture(cfg),
                intermediate_frame_visualizers=self._build_list(
                    PluginCategory.VISUALIZER,
                    self._intermediate_frame_visualizers(cfg),
                    "pipeline.intermediate_frames.visualizers",
                ),
                stream_runtime=self._stream_runtime(cfg),
                source_config=versioned_config.source_config(),
            )
        except PipelineConfigurationError:
            raise
        except Exception as exc:
            raise PipelineConfigurationError(f"Invalid pipeline configuration: {exc}", cause=exc) from exc

    # ── Internals ────────────────────────────────────────────────────────────

    def _create(self, category: PluginCategory, cfg: dict[str, Any], path: str) -> Any:
        name = self._required_string(cfg, f"{path}.name")
        params = cfg.get("params", {})
        if not isinstance(params, dict):
            raise ConfigSchemaError(f"'{path}.params' must be a mapping.", path=f"{path}.params")
        try:
            return self._registry.create(category, name, **params)
        except KeyError as exc:
            available = self._registry.available_names(category, include_aliases=True)
            raise PluginResolutionError(
                category=str(category),
                name=name,
                path=path,
                available=available,
                cause=exc,
            ) from exc
        except TypeError as exc:
            raise PluginConstructionError(
                category=str(category),
                name=name,
                path=path,
                cause=exc,
                invalid_params=True,
            ) from exc
        except Exception as exc:
            raise PluginConstructionError(
                category=str(category),
                name=name,
                path=path,
                cause=exc,
            ) from exc

    def _frame_processors(self, cfg: dict[str, Any]) -> list[IFrameBufferProcessor]:
        entries = self._optional_list(cfg, "frame_processors", "pipeline.frame_processors")
        processors: list[IFrameBufferProcessor] = []
        for index, item in enumerate(entries):
            path = f"pipeline.frame_processors[{index}]"
            entry = self._ensure_mapping(item, path)
            processor_type = str(entry.get("processor_type", "single_frame"))
            if processor_type == "single_frame":
                processors.append(SingleFrameProcessorAdapter(self._create(PluginCategory.SINGLE_FRAME_PROCESSOR, entry, path)))
            elif processor_type == "frame_buffer":
                processors.append(self._create(PluginCategory.FRAME_BUFFER_PROCESSOR, entry, path))
            else:
                raise ConfigSchemaError(
                    f"'{path}.processor_type' must be 'single_frame' or 'frame_buffer'.",
                    path=f"{path}.processor_type",
                )
        return processors

    def _build_list(
        self,
        category: PluginCategory,
        cfgs: list[dict[str, Any]],
        path: str,
    ) -> list:
        return [self._create(category, self._ensure_mapping(item, f"{path}[{index}]"), f"{path}[{index}]") for index, item in enumerate(cfgs)]

    def _unbound_visualizers(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        visualizers = self._optional_list(cfg, "visualizers", "pipeline.visualizers")
        return [
            self._ensure_mapping(item, f"pipeline.visualizers[{index}]")
            for index, item in enumerate(visualizers)
            if "result_indices" not in self._ensure_mapping(item, f"pipeline.visualizers[{index}]")
        ]

    def _visualizer_bindings(self, cfg: dict[str, Any]) -> list[VisualizerBinding]:
        bindings: list[VisualizerBinding] = []
        visualizer_configs = self._optional_list(cfg, "visualizers", "pipeline.visualizers")
        for index, item in enumerate(visualizer_configs):
            path = f"pipeline.visualizers[{index}]"
            if "result_indices" not in item:
                continue
            visualizer = self._create(
                PluginCategory.VISUALIZER,
                self._ensure_mapping(item, path),
                path,
            )
            bindings.append(
                VisualizerBinding(
                    visualizer=visualizer,
                    result_indices=self._result_indices(item, f"{path}.result_indices"),
                )
            )
        return bindings

    def _intermediate_frame_capture(self, cfg: dict[str, Any]) -> IntermediateFrameCaptureConfig:
        section = cfg.get("intermediate_frames")
        if section is None:
            if self._frame_processor_debug_capture_requested(cfg):
                return IntermediateFrameCaptureConfig.from_mapping({"enabled": True})
            return IntermediateFrameCaptureConfig.disabled()
        if not isinstance(section, dict):
            raise ConfigSchemaError(
                "'pipeline.intermediate_frames' must be a mapping.",
                path="pipeline.intermediate_frames",
            )
        capture_config = {key: value for key, value in section.items() if key != "visualizers"}
        if not capture_config and section.get("visualizers"):
            capture_config["enabled"] = True
        return IntermediateFrameCaptureConfig.from_mapping(capture_config)

    @staticmethod
    def _stream_runtime(cfg: dict[str, Any]) -> StreamRuntimeConfig:
        return StreamRuntimeConfig.from_mapping(cfg.get("runtime"))

    @staticmethod
    def _frame_processor_debug_capture_requested(cfg: dict[str, Any]) -> bool:
        """
        Enable capture when a processor explicitly asks to emit debug frames.

        Processor-level emit flags are otherwise impossible to observe because
        SingleFrameProcessorAdapter only calls emitters when the intermediate
        capture store is enabled.
        """
        frame_processors = cfg.get("frame_processors", [])
        if not isinstance(frame_processors, list):
            return False

        for item in frame_processors:
            if not isinstance(item, Mapping):
                continue
            params = item.get("params", {})
            if not isinstance(params, Mapping):
                continue
            if ConfigPipelineBuilder._debug_emit_flags_enabled(params):
                return True
            nested_config = params.get("config", {})
            if isinstance(nested_config, Mapping) and ConfigPipelineBuilder._debug_emit_flags_enabled(nested_config):
                return True
        return False

    @staticmethod
    def _debug_emit_flags_enabled(params: Mapping[str, Any]) -> bool:
        return bool(params.get("emit_intermediate_artifacts")) or bool(params.get("emit_comparison_overlay"))

    def _intermediate_frame_visualizers(self, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        section = cfg.get("intermediate_frames")
        if section is None:
            return []
        if not isinstance(section, dict):
            raise ConfigSchemaError(
                "'pipeline.intermediate_frames' must be a mapping.",
                path="pipeline.intermediate_frames",
            )
        value = section.get("visualizers", [])
        if not isinstance(value, list):
            raise ConfigSchemaError(
                "'pipeline.intermediate_frames.visualizers' must be a list.",
                path="pipeline.intermediate_frames.visualizers",
            )
        return value

    @staticmethod
    def _result_indices(config: dict[str, Any], path: str) -> tuple[int, ...]:
        value = config.get("result_indices")
        if not isinstance(value, list):
            raise ConfigSchemaError(f"'{path}' must be a list of non-negative integers.", path=path)
        if any(not isinstance(index, int) or index < 0 for index in value):
            raise ConfigSchemaError(f"'{path}' must contain only non-negative integers.", path=path)
        return tuple(value)

    @staticmethod
    def _required_mapping(
        config: dict[str, Any],
        key: str,
        path: str | None = None,
    ) -> dict[str, Any]:
        display_path = path or key
        if key not in config:
            raise ConfigSchemaError(f"Missing required config section '{display_path}'.", path=display_path)
        return ConfigPipelineBuilder._ensure_mapping(config[key], display_path)

    @staticmethod
    def _required_list(
        config: dict[str, Any],
        key: str,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        display_path = path or key
        if key not in config:
            raise ConfigSchemaError(f"Missing required config section '{display_path}'.", path=display_path)
        value = config[key]
        if not isinstance(value, list):
            raise ConfigSchemaError(f"'{display_path}' must be a list.", path=display_path)
        return value

    @staticmethod
    def _optional_list(
        config: dict[str, Any],
        key: str,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        display_path = path or key
        value = config.get(key, [])
        if not isinstance(value, list):
            raise ConfigSchemaError(f"'{display_path}' must be a list.", path=display_path)
        return value

    @staticmethod
    def _required_string(config: dict[str, Any], path: str) -> str:
        key = path.rsplit(".", maxsplit=1)[-1]
        if key not in config:
            raise ConfigSchemaError(f"Missing required config section '{path}'.", path=path)
        value = config[key]
        if not isinstance(value, str) or not value:
            raise ConfigSchemaError(f"'{path}' must be a non-empty string.", path=path)
        return value

    @staticmethod
    def _ensure_mapping(value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ConfigSchemaError(f"'{path}' must be a mapping.", path=path)
        return dict(value)
