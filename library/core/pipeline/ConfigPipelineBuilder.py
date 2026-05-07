from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from library.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameCaptureConfig
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineErrors import PipelineConfigurationError
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.core.pipeline.VisualizerBinding import VisualizerBinding
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry


class ConfigPipelineBuilder:
    """
    Declarative pipeline builder driven by a configuration dictionary.

    Design rationale
    ----------------
    ConfigPipelineBuilder exists alongside FluentPipelineBuilder to
    support deployment scenarios where the pipeline topology is defined
    externally (YAML, JSON, UI form) rather than in Python code.

    The builder delegates ALL instantiation to the PluginRegistry so that
    it never imports concrete implementations directly — it only knows
    about categories and names defined in PluginCategory.

    Configuration schema
    --------------------
    pipeline:
      frame_extractor:                  # required
        name: opencv_buffered
        params:
          source: "/path/to/video.mp4"

      frame_processors:
        - name: opencv_gray             # default processor_type: single_frame
          processor_type: single_frame
        - name: motion_magnification
          processor_type: frame_buffer

      signal_extractor:                 # required
        name: opencv_tracker
        params:
          roi: [100, 200, 50, 80]

      signal_cleaners:                  # optional list
        - name: moving_average
          params:
            window: 5

      analyzers:                        # required list (min 1)
        - name: vertical_position

      visualizers:                      # optional list
        - name: matplotlib
          result_indices: [0]            # optional; omit to visualize all results

      intermediate_frames:              # optional frame-processing debug stream
        enabled: true
        sampling_interval: 10
        max_stored_frames: 20
        export_directory: artifacts/debug
        lazy_saving: true
        visualizers:
          - name: intermediate_frames_grid

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

    def build_context(self, config: dict) -> PipelineContext:
        cfg = self._required_mapping(config, "pipeline")

        try:
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
                source_config={"pipeline": deepcopy(cfg)},
            )
        except PipelineConfigurationError:
            raise
        except Exception as exc:
            raise PipelineConfigurationError(f"Invalid pipeline configuration: {exc}") from exc

    # ── Internals ────────────────────────────────────────────────────────────

    def _create(self, category: PluginCategory, cfg: dict[str, Any], path: str) -> Any:
        name = self._required_string(cfg, f"{path}.name")
        params = cfg.get("params", {})
        if not isinstance(params, dict):
            raise PipelineConfigurationError(f"'{path}.params' must be a mapping.")
        try:
            return self._registry.create(category, name, **params)
        except KeyError as exc:
            raise PipelineConfigurationError(f"Unknown plugin '{name}' for '{path}' in category '{category}'.") from exc
        except TypeError as exc:
            raise PipelineConfigurationError(f"Invalid params for plugin '{name}' at '{path}': {exc}") from exc
        except Exception as exc:
            raise PipelineConfigurationError(f"Failed to create plugin '{name}' at '{path}': {exc}") from exc

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
                raise PipelineConfigurationError(
                    f"'{path}.processor_type' must be 'single_frame' or 'frame_buffer'."
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
        return [
            item
            for item in self._optional_list(cfg, "visualizers", "pipeline.visualizers")
            if "result_indices" not in item
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
            raise PipelineConfigurationError("'pipeline.intermediate_frames' must be a mapping.")
        capture_config = {key: value for key, value in section.items() if key != "visualizers"}
        if not capture_config and section.get("visualizers"):
            capture_config["enabled"] = True
        return IntermediateFrameCaptureConfig.from_mapping(capture_config)

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
            raise PipelineConfigurationError("'pipeline.intermediate_frames' must be a mapping.")
        value = section.get("visualizers", [])
        if not isinstance(value, list):
            raise PipelineConfigurationError("'pipeline.intermediate_frames.visualizers' must be a list.")
        return value

    @staticmethod
    def _result_indices(config: dict[str, Any], path: str) -> tuple[int, ...]:
        value = config.get("result_indices")
        if not isinstance(value, list):
            raise PipelineConfigurationError(f"'{path}' must be a list of non-negative integers.")
        if any(not isinstance(index, int) or index < 0 for index in value):
            raise PipelineConfigurationError(f"'{path}' must contain only non-negative integers.")
        return tuple(value)

    @staticmethod
    def _required_mapping(
        config: dict[str, Any],
        key: str,
        path: str | None = None,
    ) -> dict[str, Any]:
        display_path = path or key
        if key not in config:
            raise PipelineConfigurationError(f"Missing required config section '{display_path}'.")
        return ConfigPipelineBuilder._ensure_mapping(config[key], display_path)

    @staticmethod
    def _required_list(
        config: dict[str, Any],
        key: str,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        display_path = path or key
        if key not in config:
            raise PipelineConfigurationError(f"Missing required config section '{display_path}'.")
        value = config[key]
        if not isinstance(value, list):
            raise PipelineConfigurationError(f"'{display_path}' must be a list.")
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
            raise PipelineConfigurationError(f"'{display_path}' must be a list.")
        return value

    @staticmethod
    def _required_string(config: dict[str, Any], path: str) -> str:
        key = path.rsplit(".", maxsplit=1)[-1]
        if key not in config:
            raise PipelineConfigurationError(f"Missing required config section '{path}'.")
        value = config[key]
        if not isinstance(value, str) or not value:
            raise PipelineConfigurationError(f"'{path}' must be a non-empty string.")
        return value

    @staticmethod
    def _ensure_mapping(value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise PipelineConfigurationError(f"'{path}' must be a mapping.")
        return value
