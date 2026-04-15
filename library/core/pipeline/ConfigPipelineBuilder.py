from __future__ import annotations

from typing import Any

from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineErrors import PipelineConfigurationError
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

      frame_cleaners:                   # optional list
        - name: opencv_gray

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
                frame_cleaners=self._build_list(
                    PluginCategory.FRAME_CLEANER,
                    self._optional_list(cfg, "frame_cleaners", "pipeline.frame_cleaners"),
                    "pipeline.frame_cleaners",
                ),
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
