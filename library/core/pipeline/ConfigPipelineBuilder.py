from __future__ import annotations

from typing import Any

from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.plugins.PluginRegistry import PluginRegistry, PluginCategory


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

      orchestration:                    # optional orchestration settings
        max_retries: 2

    Raises
    ------
    KeyError
        If a required section is missing from the config dict.
    ValueError
        If a plugin entry is missing its 'name' field.
    """

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    # ── Public API ───────────────────────────────────────────────────────────

    def build(self, config: dict) -> PipelineOrchestrator:
        """
        Build and return a fully configured PipelineOrchestrator from *config*.

        max_retries is read from config['pipeline']['orchestration']['max_retries']
        and defaults to 0 if the key is absent.
        """
        context = self._build_context(config)
        max_retries = config.get("pipeline", {}).get("orchestration", {}).get("max_retries", 0)
        return PipelineOrchestrator(context, max_retries=max_retries)

    # ── Internals ────────────────────────────────────────────────────────────

    def _build_context(self, config: dict) -> PipelineContext:
        cfg = config["pipeline"]

        return PipelineContext(
            frame_extractor  = self._create(PluginCategory.FRAME_EXTRACTOR,
                                            cfg["frame_extractor"]),
            signal_extractor = self._create(PluginCategory.SIGNAL_EXTRACTOR,
                                            cfg["signal_extractor"]),
            frame_cleaners   = self._build_list(PluginCategory.FRAME_CLEANER,
                                                cfg.get("frame_cleaners", [])),
            signal_cleaners  = self._build_list(PluginCategory.SIGNAL_CLEANER,
                                                cfg.get("signal_cleaners", [])),
            analyzers        = self._build_list(PluginCategory.ANALYZER,
                                                cfg["analyzers"]),
            visualizers      = self._build_list(PluginCategory.VISUALIZER,
                                                cfg.get("visualizers", [])),
        )

    def _create(self, category: PluginCategory, cfg: dict[str, Any]) -> Any:
        if "name" not in cfg:
            raise ValueError(
                f"ConfigPipelineBuilder: missing 'name' in config for category '{category}'."
            )
        return self._registry.create(category, cfg["name"], **cfg.get("params", {}))

    def _build_list(self, category: PluginCategory, cfgs: list[dict]) -> list:
        return [self._create(category, c) for c in cfgs]