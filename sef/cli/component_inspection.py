from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Mapping
from typing import Any

from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.pipeline.PipelineExportUtils import yaml_dumps
from sef.core.plugins import PluginCategory, PluginDefinition, PluginRegistry


def component_descriptors(descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort component descriptors for stable CLI output."""
    return sorted(descriptors, key=lambda item: (str(item["category"]), str(item["name"])))


def find_component_matches(
    registry: PluginRegistry,
    name: str,
    *,
    category: str | None = None,
) -> list[PluginDefinition]:
    """Find unique plugin definitions matching a name or alias."""
    categories = (category,) if category else tuple(category.value for category in PluginCategory)
    matches: list[PluginDefinition] = []
    for category_name in categories:
        try:
            matches.append(registry.get(category_name, name))
        except KeyError:
            continue
    unique: dict[tuple[str, str], PluginDefinition] = {}
    for definition in matches:
        unique[(definition.category, definition.name)] = definition
    return list(unique.values())


def component_inspection(definition: PluginDefinition) -> dict[str, Any]:
    """Build the detailed component inspection payload."""
    signature_info = _signature_info(definition)
    capabilities = _definition_capabilities(definition)
    payload = {
        **definition.as_dict(),
        "signature": signature_info["signature"],
        "required_params": signature_info["required_params"],
        "optional_params": signature_info["optional_params"],
        "var_params": signature_info["var_params"],
        "capabilities": capabilities.as_dict() if capabilities else None,
        "streaming": bool(capabilities.supports_streaming) if capabilities else None,
        "input": _category_input(definition.category),
        "output": _category_output(definition.category),
    }
    payload["yaml_example"] = _yaml_component_snippet(definition)
    payload["python_example"] = _python_component_snippet(definition)
    return payload


def print_component_inspection(payload: Mapping[str, Any]) -> None:
    """Print the detailed component inspection payload."""
    print(f"name: {payload['name']}")
    print(f"category: {payload['category']}")
    print(f"description: {payload.get('description') or '-'}")
    print(f"version: {payload.get('version')}")
    print(f"aliases: {', '.join(payload.get('aliases') or []) or '-'}")
    print(f"factory: {payload.get('factory_path')}")
    print(f"signature: {payload.get('signature')}")
    print(f"required_params: {', '.join(payload.get('required_params') or []) or '-'}")
    optional = payload.get("optional_params") or {}
    if optional:
        print("optional_params:")
        for name, default in optional.items():
            print(f"  - {name}={default}")
    else:
        print("optional_params: -")
    capabilities = payload.get("capabilities")
    if capabilities:
        print("capabilities:")
        for key, value in capabilities.items():
            print(f"  - {key}: {value}")
    else:
        print("capabilities: unknown without instantiation")
    print(f"input: {payload.get('input')}")
    print(f"output: {payload.get('output')}")
    _print_metadata(payload.get("metadata") or {})
    print("yaml_example:")
    print(payload["yaml_example"])
    print("python_example:")
    print(payload["python_example"])


def _print_metadata(metadata: Mapping[str, Any]) -> None:
    if not metadata:
        print("metadata: -")
        return
    print("metadata:")
    for key in sorted(metadata):
        print(f"  - {key}: {_format_metadata_value(metadata[key])}")


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float) or value is None:
        return str(value)
    return json.dumps(value, sort_keys=True)


def _signature_info(definition: PluginDefinition) -> dict[str, Any]:
    target = _signature_target(definition)
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return {"signature": "unknown", "required_params": [], "optional_params": {}, "var_params": []}

    parameters = list(signature.parameters.values())
    parameters = _drop_non_config_parameters(parameters, definition.category)
    required: list[str] = []
    optional: dict[str, str] = {}
    var_params: list[str] = []
    for parameter in parameters:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            var_params.append(parameter.name)
            continue
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
        else:
            optional[parameter.name] = repr(parameter.default)
    return {
        "signature": str(signature),
        "required_params": required,
        "optional_params": optional,
        "var_params": var_params,
    }


def _signature_target(definition: PluginDefinition) -> Any:
    factory = definition.factory
    if inspect.isclass(factory):
        return factory.__init__
    original = _closure_callable(factory)
    if original is not None:
        return original
    metadata_function = definition.metadata.get("function")
    if isinstance(metadata_function, str):
        resolved = _resolve_dotted_path(metadata_function)
        if resolved is not None:
            return resolved
    return factory


def _closure_callable(factory: Any) -> Any | None:
    closure = getattr(factory, "__closure__", None) or ()
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value) and value is not factory:
            return value
    return None


def _resolve_dotted_path(path: str) -> Any | None:
    module_name, _, attr_path = path.partition(":") if ":" in path else path.rpartition(".")
    if not module_name or not attr_path:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    target: Any = module
    for part in attr_path.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target


def _drop_non_config_parameters(
    parameters: list[inspect.Parameter],
    category: str,
) -> list[inspect.Parameter]:
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    if not parameters:
        return parameters
    input_first_categories = {
        PluginCategory.SINGLE_FRAME_PROCESSOR.value,
        PluginCategory.SIGNAL_EXTRACTOR.value,
        PluginCategory.SIGNAL_CLEANER.value,
        PluginCategory.ANALYZER.value,
        PluginCategory.VISUALIZER.value,
    }
    if category in input_first_categories and parameters[0].name not in {"config", "params"}:
        parameters = parameters[1:]
    return [parameter for parameter in parameters if parameter.name != "context"]


def _definition_capabilities(definition: PluginDefinition) -> StageCapabilities | None:
    factory = definition.factory
    capabilities = getattr(factory, "capabilities", None)
    if isinstance(capabilities, StageCapabilities):
        return capabilities
    return None


def _category_input(category: str) -> str:
    return {
        PluginCategory.FRAME_EXTRACTOR.value: "none",
        PluginCategory.SINGLE_FRAME_PROCESSOR.value: "Frame or frame image",
        PluginCategory.FRAME_BUFFER_PROCESSOR.value: "FrameBuffer",
        PluginCategory.SIGNAL_EXTRACTOR.value: "FrameBuffer",
        PluginCategory.SIGNAL_CLEANER.value: "ISignal",
        PluginCategory.ANALYZER.value: "ISignal",
        PluginCategory.VISUALIZER.value: "IData analysis result",
        PluginCategory.BRANCHING_RULE.value: "Pipeline trigger event",
    }.get(category, "unknown")


def _category_output(category: str) -> str:
    return {
        PluginCategory.FRAME_EXTRACTOR.value: "FrameBuffer",
        PluginCategory.SINGLE_FRAME_PROCESSOR.value: "Frame",
        PluginCategory.FRAME_BUFFER_PROCESSOR.value: "FrameBuffer or FrameExportResult",
        PluginCategory.SIGNAL_EXTRACTOR.value: "ISignal",
        PluginCategory.SIGNAL_CLEANER.value: "ISignal",
        PluginCategory.ANALYZER.value: "IData",
        PluginCategory.VISUALIZER.value: "tuple[VisualArtifact, ...]",
        PluginCategory.BRANCHING_RULE.value: "branch decision",
    }.get(category, "unknown")


def _yaml_component_snippet(definition: PluginDefinition) -> str:
    params = _example_params(definition)
    body: dict[str, Any] = {"name": definition.name}
    if params:
        body["params"] = params
    if definition.category in {
        PluginCategory.SINGLE_FRAME_PROCESSOR.value,
        PluginCategory.FRAME_BUFFER_PROCESSOR.value,
    }:
        body["processor_type"] = "frame_buffer" if definition.category == PluginCategory.FRAME_BUFFER_PROCESSOR.value else "single_frame"
    return yaml_dumps(body).rstrip()


def _python_component_snippet(definition: PluginDefinition) -> str:
    method = {
        PluginCategory.FRAME_EXTRACTOR.value: "frames",
        PluginCategory.SINGLE_FRAME_PROCESSOR.value: "process",
        PluginCategory.FRAME_BUFFER_PROCESSOR.value: "process",
        PluginCategory.SIGNAL_EXTRACTOR.value: "signals",
        PluginCategory.SIGNAL_CLEANER.value: "clean",
        PluginCategory.ANALYZER.value: "analyze",
        PluginCategory.VISUALIZER.value: "visualize",
    }.get(definition.category, "register")
    params = _example_params(definition)
    if params:
        formatted = ", ".join(f"{key}=..." for key in params)
        return f"sef.pipeline(include_builtins=True).{method}({definition.name!r}, {formatted})"
    return f"sef.pipeline(include_builtins=True).{method}({definition.name!r})"


def _example_params(definition: PluginDefinition) -> dict[str, str]:
    info = _signature_info(definition)
    return {name: "<required>" for name in info["required_params"] if name not in {"config", "buffer"}}
