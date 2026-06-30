from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sef.api import normalize_config
from sef.cli.constants import UNKNOWN_FIELD_DOC
from sef.cli.diagnostics import DiagnosticItem
from sef.core.errors import ConfigSchemaError
from sef.core.pipeline.PipelineExportUtils import yaml_dumps
from sef.core.pipeline.PipelineRunOptions import (
    RUN_OPTIONS_CONFIG_KEY,
    RUN_OPTIONS_EXECUTION_PLAN_CONFIG_KEY,
    RUN_OPTIONS_REPRODUCIBILITY_CONFIG_KEY,
)


@dataclass(frozen=True, slots=True)
class ConfigInspectionResult:
    """Result of lightweight CLI schema inspection."""

    warnings: tuple[DiagnosticItem, ...]
    errors: tuple[DiagnosticItem, ...]

    @property
    def has_errors(self) -> bool:
        """Return True when strict inspection found blocking issues."""
        return bool(self.errors)


class ConfigInspector:
    """Performs practical CLI config checks before the core builder runs."""

    _TOP_LEVEL_KEYS = frozenset({"schema_version", "pipeline", RUN_OPTIONS_CONFIG_KEY})
    _PIPELINE_KEYS = frozenset(
        {
            "frame_extractor",
            "frame_processors",
            "signal_extractor",
            "signal_cleaners",
            "analyzers",
            "visualizers",
            "intermediate_frames",
            "runtime",
        }
    )
    _COMPONENT_KEYS = frozenset({"name", "params", "processor_type", "result_indices"})
    _INTERMEDIATE_FRAME_KEYS = frozenset(
        {
            "enabled",
            "sampling_interval",
            "max_stored_frames",
            "export_directory",
            "lazy_saving",
            "visualizers",
        }
    )
    _RUNTIME_KEYS = frozenset({"frame_buffer_size", "signal_buffer_size", "data_buffer_size", "latency_policy"})
    _LATENCY_POLICY_KEYS = frozenset({"name", "params"})
    _RUN_OPTIONS_KEYS = frozenset(
        {
            RUN_OPTIONS_EXECUTION_PLAN_CONFIG_KEY,
            RUN_OPTIONS_REPRODUCIBILITY_CONFIG_KEY,
        }
    )

    def inspect(self, config: Mapping[str, Any], *, strict: bool = False) -> ConfigInspectionResult:
        """Return unknown-field diagnostics for the public config shape."""
        warnings: list[DiagnosticItem] = []
        errors: list[DiagnosticItem] = []

        self._unknown_keys(config, self._TOP_LEVEL_KEYS, "config", strict, warnings, errors)
        pipeline = config.get("pipeline")
        if isinstance(pipeline, Mapping):
            self._inspect_pipeline(pipeline, strict, warnings, errors)
        run_options = config.get(RUN_OPTIONS_CONFIG_KEY)
        if isinstance(run_options, Mapping):
            self._unknown_keys(run_options, self._RUN_OPTIONS_KEYS, RUN_OPTIONS_CONFIG_KEY, strict, warnings, errors)

        return ConfigInspectionResult(tuple(warnings), tuple(errors))

    def _inspect_pipeline(
        self,
        pipeline: Mapping[str, Any],
        strict: bool,
        warnings: list[DiagnosticItem],
        errors: list[DiagnosticItem],
    ) -> None:
        self._unknown_keys(pipeline, self._PIPELINE_KEYS, "pipeline", strict, warnings, errors)
        self._inspect_component(pipeline.get("frame_extractor"), "pipeline.frame_extractor", strict, warnings, errors)
        self._inspect_component(pipeline.get("signal_extractor"), "pipeline.signal_extractor", strict, warnings, errors)
        self._inspect_component_list(pipeline.get("frame_processors", ()), "pipeline.frame_processors", strict, warnings, errors)
        self._inspect_component_list(pipeline.get("signal_cleaners", ()), "pipeline.signal_cleaners", strict, warnings, errors)
        self._inspect_component_list(pipeline.get("analyzers", ()), "pipeline.analyzers", strict, warnings, errors)
        self._inspect_component_list(pipeline.get("visualizers", ()), "pipeline.visualizers", strict, warnings, errors)
        self._inspect_intermediate_frames(pipeline.get("intermediate_frames"), strict, warnings, errors)
        self._inspect_runtime(pipeline.get("runtime"), strict, warnings, errors)

    def _inspect_component_list(
        self,
        value: Any,
        path: str,
        strict: bool,
        warnings: list[DiagnosticItem],
        errors: list[DiagnosticItem],
    ) -> None:
        if not isinstance(value, list):
            return
        for index, item in enumerate(value):
            self._inspect_component(item, f"{path}[{index}]", strict, warnings, errors)

    def _inspect_component(
        self,
        value: Any,
        path: str,
        strict: bool,
        warnings: list[DiagnosticItem],
        errors: list[DiagnosticItem],
    ) -> None:
        if not isinstance(value, Mapping):
            return
        self._unknown_keys(value, self._COMPONENT_KEYS, path, strict, warnings, errors)

    def _inspect_intermediate_frames(
        self,
        value: Any,
        strict: bool,
        warnings: list[DiagnosticItem],
        errors: list[DiagnosticItem],
    ) -> None:
        if not isinstance(value, Mapping):
            return
        self._unknown_keys(value, self._INTERMEDIATE_FRAME_KEYS, "pipeline.intermediate_frames", strict, warnings, errors)
        self._inspect_component_list(
            value.get("visualizers", ()),
            "pipeline.intermediate_frames.visualizers",
            strict,
            warnings,
            errors,
        )

    def _inspect_runtime(
        self,
        value: Any,
        strict: bool,
        warnings: list[DiagnosticItem],
        errors: list[DiagnosticItem],
    ) -> None:
        if not isinstance(value, Mapping):
            return
        self._unknown_keys(value, self._RUNTIME_KEYS, "pipeline.runtime", strict, warnings, errors)
        latency_policy = value.get("latency_policy")
        if isinstance(latency_policy, Mapping):
            self._unknown_keys(
                latency_policy,
                self._LATENCY_POLICY_KEYS,
                "pipeline.runtime.latency_policy",
                strict,
                warnings,
                errors,
            )

    @staticmethod
    def _unknown_keys(
        value: Mapping[str, Any],
        allowed: frozenset[str],
        path: str,
        strict: bool,
        warnings: list[DiagnosticItem],
        errors: list[DiagnosticItem],
    ) -> None:
        for key in sorted(str(item) for item in value.keys()):
            if key in allowed:
                continue
            item = DiagnosticItem(
                "error" if strict else "warning",
                f"Unknown field `{path}.{key}`.",
                cause="This field is not part of the current public SEF config schema.",
                suggestion=UNKNOWN_FIELD_DOC,
            )
            if strict:
                errors.append(item)
            else:
                warnings.append(item)


def load_config_with_raw(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load JSON/YAML config preserving both raw and normalized forms."""
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        raw = _load_yaml_raw(raw_text, config_path)
    else:
        raise ConfigSchemaError(
            f"Unsupported config file extension '{config_path.suffix}'. Use .json, .yaml, or .yml.",
            path=str(config_path),
        )
    if not isinstance(raw, Mapping):
        raise ConfigSchemaError("'config' must be a mapping.", path=str(config_path))
    return dict(raw), normalize_config(raw)


def inspect_cli_config(
    raw_config: Mapping[str, Any],
    normalized_config: Mapping[str, Any],
    *,
    strict: bool,
) -> ConfigInspectionResult:
    """Inspect user-facing config fields after normalization."""
    config_to_inspect = raw_config if "pipeline" in raw_config else normalized_config
    return ConfigInspector().inspect(config_to_inspect, strict=strict)


def public_config_schema() -> dict[str, Any]:
    """Return the practical public config schema printed by `sef config schema`."""
    return {
        "type": "object",
        "required": ["schema_version", "pipeline"],
        "properties": {
            "schema_version": {"type": "string", "example": "1.0"},
            RUN_OPTIONS_CONFIG_KEY: {
                "type": "object",
                "description": "Optional run metadata controls. Omit for lowest-overhead execution.",
                "properties": {
                    RUN_OPTIONS_EXECUTION_PLAN_CONFIG_KEY: {
                        "oneOf": [
                            {"type": "boolean"},
                            {"enum": ["none", "summary", "full"]},
                        ],
                        "description": "Optional execution-plan metadata. true maps to full; false maps to none.",
                    },
                    RUN_OPTIONS_REPRODUCIBILITY_CONFIG_KEY: {"type": "boolean", "default": False},
                },
            },
            "pipeline": {
                "type": "object",
                "required": ["frame_extractor", "signal_extractor", "analyzers"],
                "properties": {
                    "frame_extractor": _component_schema("frame_extractor"),
                    "frame_processors": {
                        "type": "array",
                        "items": {
                            **_component_schema("frame_processor"),
                            "properties": {
                                "name": {"type": "string"},
                                "params": {"type": "object"},
                                "processor_type": {"enum": ["single_frame", "frame_buffer"], "default": "single_frame"},
                            },
                        },
                    },
                    "signal_extractor": _component_schema("signal_extractor"),
                    "signal_cleaners": {"type": "array", "items": _component_schema("signal_cleaner")},
                    "analyzers": {"type": "array", "items": _component_schema("analyzer")},
                    "visualizers": {
                        "type": "array",
                        "items": {
                            **_component_schema("visualizer"),
                            "properties": {
                                "name": {"type": "string"},
                                "params": {"type": "object"},
                                "result_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                            },
                        },
                    },
                    "intermediate_frames": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "default": False},
                            "sampling_interval": {"type": "integer", "minimum": 1},
                            "max_stored_frames": {"type": "integer", "minimum": 1},
                            "export_directory": {"type": "string"},
                            "lazy_saving": {"type": "boolean"},
                            "visualizers": {"type": "array", "items": _component_schema("visualizer")},
                        },
                    },
                    "runtime": {
                        "type": "object",
                        "properties": {
                            "frame_buffer_size": {"type": "integer", "minimum": 1},
                            "signal_buffer_size": {"type": "integer", "minimum": 1},
                            "data_buffer_size": {"type": "integer", "minimum": 1},
                            "latency_policy": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "example": "blocking"},
                                    "params": {"type": "object"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def dump_config_schema(schema: Mapping[str, Any], *, output_format: str) -> str:
    """Serialize a config schema for CLI output."""
    if output_format == "yaml":
        return yaml_dumps(schema)
    return json.dumps(schema, indent=2, sort_keys=False)


def _load_yaml_raw(raw_text: str, config_path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigSchemaError(
            "YAML config loading requires PyYAML. Install SEF with its declared runtime dependencies.",
            path=str(config_path),
            cause=exc,
        ) from exc
    return yaml.safe_load(raw_text)


def _component_schema(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "params": {"type": "object", "additionalProperties": True},
        },
    }
