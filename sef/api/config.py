from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from library.core.pipeline.PipelineConfigVersioning import normalize_pipeline_config
from library.core.pipeline.PipelineErrors import ConfigSchemaError

_PIPELINE_STAGE_KEYS = frozenset(
    {
        "frame_extractor",
        "frame_processors",
        "frame_cleaners",
        "signal_extractor",
        "signal_cleaners",
        "analyzers",
        "visualizers",
        "intermediate_frames",
        "runtime",
    }
)
_FRAME_SOURCE_CONFIG_KEYS = ("resize", "stride", "max_frames")
_TRACKING_SIGNAL_EXTRACTORS = frozenset(
    {
        "opencv_tracker",
        "opencv_stream_tracker",
        "opencv_multi_tracker",
    }
)


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a SEF pipeline config from JSON or YAML and return the canonical schema.

    The returned mapping is safe to pass to ``sef.from_config`` or
    ``ConfigPipelineBuilder``. YAML support intentionally lives here instead of
    in the execution core so the runtime remains independent from file formats.
    """
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        loaded = _load_yaml(raw_text, config_path)
    else:
        raise ConfigSchemaError(
            f"Unsupported config file extension '{config_path.suffix}'. Use .json, .yaml, or .yml.",
            path=str(config_path),
        )
    return normalize_config(loaded)


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize user-facing config shapes into the public SEF schema.

    This is the single compatibility entry point for CLI, UI adapters, and
    facade-based config execution. It accepts both a full top-level config and a
    bare pipeline section, then applies public schema version normalization.
    """
    root = _ensure_root_config(config)
    versioned = normalize_pipeline_config(root).source_config()
    pipeline = dict(versioned["pipeline"])
    _move_frame_source_options(pipeline)
    _inject_frame_source_path(pipeline)
    versioned["pipeline"] = pipeline
    return versioned


def _load_yaml(raw_text: str, config_path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only in broken installs
        raise ConfigSchemaError(
            "YAML config loading requires PyYAML. Install SEF with its declared runtime dependencies.",
            path=str(config_path),
            cause=exc,
        ) from exc

    return yaml.safe_load(raw_text)


def _ensure_root_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ConfigSchemaError("'config' must be a mapping.", path="config")
    root = dict(config)
    if "pipeline" in root:
        return root
    if _PIPELINE_STAGE_KEYS.intersection(root):
        return {"pipeline": root}
    raise ConfigSchemaError("Missing required config section 'pipeline'.", path="pipeline")


def _move_frame_source_options(pipeline: dict[str, Any]) -> None:
    frame_extractor = pipeline.get("frame_extractor")
    if not isinstance(frame_extractor, Mapping):
        return

    entry = dict(frame_extractor)
    params = entry.get("params", {})
    if not isinstance(params, Mapping):
        return

    normalized_params = dict(params)
    source_config = normalized_params.get("config", {})
    normalized_source_config = dict(source_config) if isinstance(source_config, Mapping) else {}

    moved = False
    for key in _FRAME_SOURCE_CONFIG_KEYS:
        if key in normalized_params:
            normalized_source_config[key] = normalized_params.pop(key)
            moved = True

    if moved:
        normalized_params["config"] = normalized_source_config
        entry["params"] = normalized_params
        pipeline["frame_extractor"] = entry


def _inject_frame_source_path(pipeline: dict[str, Any]) -> None:
    frame_source_path = _frame_source_path(pipeline)
    if frame_source_path is None:
        return

    signal_extractor = pipeline.get("signal_extractor")
    if not isinstance(signal_extractor, Mapping):
        return
    extractor_name = str(signal_extractor.get("name", ""))
    if extractor_name not in _TRACKING_SIGNAL_EXTRACTORS:
        return

    entry = dict(signal_extractor)
    params = entry.get("params", {})
    if not isinstance(params, Mapping):
        return

    normalized_params = dict(params)
    extractor_config = normalized_params.get("config", {})
    normalized_extractor_config = dict(extractor_config) if isinstance(extractor_config, Mapping) else {}
    normalized_extractor_config.setdefault("source_path", frame_source_path)
    normalized_params["config"] = normalized_extractor_config
    entry["params"] = normalized_params
    pipeline["signal_extractor"] = entry


def _frame_source_path(pipeline: Mapping[str, Any]) -> str | None:
    frame_extractor = pipeline.get("frame_extractor")
    if not isinstance(frame_extractor, Mapping):
        return None
    params = frame_extractor.get("params", {})
    if not isinstance(params, Mapping):
        return None
    raw_path = params.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    return raw_path
