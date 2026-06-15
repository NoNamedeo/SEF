from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sef.api.function_adapters import (
    FunctionAnalyzer,
    FunctionFrameBufferProcessor,
    FunctionFrameExtractor,
    FunctionFrameProcessor,
    FunctionSignalCleaner,
    FunctionSignalExtractor,
    FunctionVisualizer,
)
from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from sef.core.interfaces.IFrameExtractor import IFrameExtractor
from sef.core.interfaces.ISignalCleaner import ISignalCleaner
from sef.core.interfaces.ISignalExtractor import ISignalExtractor
from sef.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from sef.core.interfaces.IVisualizer import IVisualizer
from sef.core.plugins import PluginCategory, PluginRegistry

ComponentRef = str | type | object | Callable[..., Any]

_NON_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Declarative stage used by the high-level facade before config emission."""

    component: ComponentRef
    params: dict[str, Any] = field(default_factory=dict)
    processor_type: str | None = None
    result_indices: tuple[int, ...] | None = None


class StageRegistry:
    """
    Convert Python component references into registry-backed config entries.

    String references are assumed to be existing plugin names. Classes,
    instances, and plain callables are auto-registered into the scoped registry
    owned by one facade instance.
    """

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry
        self._registered_keys: dict[tuple[str, int], str] = {}

    @property
    def registry(self) -> PluginRegistry:
        """Return the scoped registry including auto-registered components."""
        return self._registry

    def entry(self, category: PluginCategory, spec: StageSpec) -> dict[str, Any]:
        name = self._name_for(category, spec)
        entry: dict[str, Any] = {"name": name}
        if spec.params:
            entry["params"] = dict(spec.params)
        if spec.result_indices is not None:
            entry["result_indices"] = list(spec.result_indices)
        if spec.processor_type is not None:
            entry["processor_type"] = spec.processor_type
        return entry

    def frame_processor_entry(self, spec: StageSpec) -> dict[str, Any]:
        processor_type = spec.processor_type or self._infer_frame_processor_type(spec.component)
        category = (
            PluginCategory.FRAME_BUFFER_PROCESSOR
            if processor_type == "frame_buffer"
            else PluginCategory.SINGLE_FRAME_PROCESSOR
        )
        resolved = StageSpec(
            component=spec.component,
            params=spec.params,
            processor_type=processor_type,
            result_indices=spec.result_indices,
        )
        return self.entry(category, resolved)

    def _name_for(self, category: PluginCategory, spec: StageSpec) -> str:
        component = spec.component
        if isinstance(component, str):
            return component

        key = (str(category), id(component))
        if key in self._registered_keys:
            return self._registered_keys[key]

        name = self._available_name(category, self._default_name(component))
        factory = self._factory_for(category, component, spec.params)
        self._registry.register(
            category,
            name,
            factory,
            f"Auto-registered by the high-level sef facade from {self._display_name(component)}.",
            metadata={"source": "sef.facade", "component": self._display_name(component)},
        )
        self._registered_keys[key] = name
        return name

    def _factory_for(
        self,
        category: PluginCategory,
        component: ComponentRef,
        params: dict[str, Any],
    ) -> Callable[..., Any]:
        if inspect.isclass(component):
            return component
        if self._is_contract_instance(category, component):
            if params:
                raise ValueError("Do not pass params when using an already constructed component instance.")
            return lambda **_: component
        if self._is_contract_like_instance(category, component):
            if params:
                raise ValueError("Do not pass params when using an already constructed component instance.")
            return lambda **_: component
        if callable(component):
            return self._function_factory(category, component)
        raise TypeError(f"Unsupported component reference for {category}: {component!r}")

    @staticmethod
    def _function_factory(category: PluginCategory, function: Callable[..., Any]) -> Callable[..., Any]:
        if category == PluginCategory.FRAME_EXTRACTOR:
            return lambda **params: FunctionFrameExtractor(function, params)
        if category == PluginCategory.SIGNAL_EXTRACTOR:
            return lambda **params: FunctionSignalExtractor(function, params)
        if category == PluginCategory.SIGNAL_CLEANER:
            return lambda **params: FunctionSignalCleaner(function, params)
        if category == PluginCategory.ANALYZER:
            return lambda **params: FunctionAnalyzer(function, params)
        if category == PluginCategory.VISUALIZER:
            return lambda **params: FunctionVisualizer(function, params)
        if category == PluginCategory.SINGLE_FRAME_PROCESSOR:
            return lambda accepts_frame=False, **params: FunctionFrameProcessor(
                function,
                params,
                accepts_frame=bool(accepts_frame),
            )
        if category == PluginCategory.FRAME_BUFFER_PROCESSOR:
            return lambda **params: FunctionFrameBufferProcessor(function, params)
        raise TypeError(f"Plain callables are not supported for {category}.")

    @staticmethod
    def _is_contract_instance(category: PluginCategory, component: object) -> bool:
        contracts = {
            PluginCategory.FRAME_EXTRACTOR: IFrameExtractor,
            PluginCategory.SIGNAL_EXTRACTOR: ISignalExtractor,
            PluginCategory.SIGNAL_CLEANER: ISignalCleaner,
            PluginCategory.ANALYZER: IAnalyzer,
            PluginCategory.VISUALIZER: IVisualizer,
            PluginCategory.SINGLE_FRAME_PROCESSOR: ISingleFrameProcessor,
            PluginCategory.FRAME_BUFFER_PROCESSOR: IFrameBufferProcessor,
        }
        contract = contracts.get(category)
        return bool(contract is not None and isinstance(component, contract))

    @staticmethod
    def _is_contract_like_instance(category: PluginCategory, component: object) -> bool:
        required_methods = {
            PluginCategory.FRAME_EXTRACTOR: "extract",
            PluginCategory.SIGNAL_EXTRACTOR: "extract",
            PluginCategory.SIGNAL_CLEANER: "clean",
            PluginCategory.ANALYZER: "analyze",
            PluginCategory.VISUALIZER: "render",
            PluginCategory.SINGLE_FRAME_PROCESSOR: "process",
            PluginCategory.FRAME_BUFFER_PROCESSOR: "process",
        }
        method_name = required_methods.get(category)
        return bool(method_name is not None and callable(getattr(component, method_name, None)))

    def _infer_frame_processor_type(self, component: ComponentRef) -> str:
        if isinstance(component, str):
            if self._registry.contains(PluginCategory.FRAME_BUFFER_PROCESSOR, component) and not self._registry.contains(
                PluginCategory.SINGLE_FRAME_PROCESSOR,
                component,
            ):
                return "frame_buffer"
            return "single_frame"
        if inspect.isclass(component) and issubclass(component, IFrameBufferProcessor):
            return "frame_buffer"
        if isinstance(component, IFrameBufferProcessor):
            return "frame_buffer"
        return "single_frame"

    def _available_name(self, category: PluginCategory, base_name: str) -> str:
        if not self._registry.contains(category, base_name):
            return base_name
        index = 2
        while self._registry.contains(category, f"{base_name}_{index}"):
            index += 1
        return f"{base_name}_{index}"

    @staticmethod
    def _default_name(component: ComponentRef) -> str:
        if inspect.isclass(component):
            raw_name = component.__name__
        else:
            raw_name = getattr(component, "__name__", type(component).__name__)
        snake = _CAMEL_RE.sub("_", raw_name).lower()
        name = _NON_IDENTIFIER_RE.sub("_", snake).strip("_.-")
        return name or "component"

    @staticmethod
    def _display_name(component: ComponentRef) -> str:
        module = getattr(component, "__module__", type(component).__module__)
        qualname = getattr(component, "__qualname__", type(component).__qualname__)
        return f"{module}.{qualname}"
