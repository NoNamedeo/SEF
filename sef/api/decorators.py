from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from sef.api.function_adapters import (
    FunctionAnalyzer,
    FunctionFrameBufferProcessor,
    FunctionFrameExtractor,
    FunctionFrameProcessor,
    FunctionSignalCleaner,
    FunctionSignalExtractor,
    FunctionVisualizer,
    resolve_function_capabilities,
)
from sef.api.registry import register_user_plugin
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.plugins import PluginCategory, PluginRegistry


def frame_extractor(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
    description: str = "",
    version: str = "1.0.0",
    aliases: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
    capabilities: StageCapabilities | None = None,
):
    """Register a function as a frame extractor."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.FRAME_EXTRACTOR,
        description=description,
        version=version,
        aliases=aliases,
        metadata=metadata,
        capabilities=capabilities,
        factory_builder=lambda function, resolved_capabilities: (
            lambda **params: FunctionFrameExtractor(function, params, capabilities=resolved_capabilities)
        ),
    )


def signal_extractor(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
    description: str = "",
    version: str = "1.0.0",
    aliases: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
    capabilities: StageCapabilities | None = None,
):
    """Register a function as a signal extractor."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.SIGNAL_EXTRACTOR,
        description=description,
        version=version,
        aliases=aliases,
        metadata=metadata,
        capabilities=capabilities,
        factory_builder=lambda function, resolved_capabilities: (
            lambda **params: FunctionSignalExtractor(function, params, capabilities=resolved_capabilities)
        ),
    )


def processor(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
    accepts_frame: bool = False,
    description: str = "",
    version: str = "1.0.0",
    aliases: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
    capabilities: StageCapabilities | None = None,
):
    """Register a function as a single-frame processor."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.SINGLE_FRAME_PROCESSOR,
        description=description,
        version=version,
        aliases=aliases,
        metadata=metadata,
        capabilities=capabilities,
        factory_builder=lambda function, resolved_capabilities: (
            lambda accepts_frame=accepts_frame, **params: FunctionFrameProcessor(
                function,
                params,
                accepts_frame=bool(accepts_frame),
                capabilities=resolved_capabilities,
            )
        ),
    )


def frame_buffer_processor(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
    description: str = "",
    version: str = "1.0.0",
    aliases: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
    capabilities: StageCapabilities | None = None,
):
    """Register a function as a frame-buffer processor."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.FRAME_BUFFER_PROCESSOR,
        description=description,
        version=version,
        aliases=aliases,
        metadata=metadata,
        capabilities=capabilities,
        factory_builder=lambda function, resolved_capabilities: (
            lambda **params: FunctionFrameBufferProcessor(function, params, capabilities=resolved_capabilities)
        ),
    )


def cleaner(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
    description: str = "",
    version: str = "1.0.0",
    aliases: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
    capabilities: StageCapabilities | None = None,
):
    """Register a function as a signal cleaner."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.SIGNAL_CLEANER,
        description=description,
        version=version,
        aliases=aliases,
        metadata=metadata,
        capabilities=capabilities,
        factory_builder=lambda function, resolved_capabilities: (
            lambda **params: FunctionSignalCleaner(function, params, capabilities=resolved_capabilities)
        ),
    )


def analyzer(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
    description: str = "",
    version: str = "1.0.0",
    aliases: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
    capabilities: StageCapabilities | None = None,
):
    """Register a function as an analyzer."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.ANALYZER,
        description=description,
        version=version,
        aliases=aliases,
        metadata=metadata,
        capabilities=capabilities,
        factory_builder=lambda function, resolved_capabilities: (
            lambda **params: FunctionAnalyzer(function, params, capabilities=resolved_capabilities)
        ),
    )


def visualizer(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
    description: str = "",
    version: str = "1.0.0",
    aliases: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
    capabilities: StageCapabilities | None = None,
):
    """Register a function as a visualizer."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.VISUALIZER,
        description=description,
        version=version,
        aliases=aliases,
        metadata=metadata,
        capabilities=capabilities,
        factory_builder=lambda function, resolved_capabilities: (
            lambda **params: FunctionVisualizer(function, params, capabilities=resolved_capabilities)
        ),
    )


def _decorator(
    name: str | Callable[..., Any] | None,
    *,
    registry: PluginRegistry | None,
    category: PluginCategory,
    description: str,
    version: str,
    aliases: Iterable[str],
    metadata: Mapping[str, Any] | None,
    capabilities: StageCapabilities | None,
    factory_builder: Callable[[Callable[..., Any], StageCapabilities | None], Callable[..., Any]],
):
    if callable(name) and not isinstance(name, str):
        function = name
        _register(
            function.__name__,
            function,
            registry,
            category,
            description=description,
            version=version,
            aliases=aliases,
            metadata=metadata,
            capabilities=capabilities,
            factory_builder=factory_builder,
        )
        return function

    def _wrap(function: Callable[..., Any]):
        plugin_name = name or function.__name__
        _register(
            str(plugin_name),
            function,
            registry,
            category,
            description=description,
            version=version,
            aliases=aliases,
            metadata=metadata,
            capabilities=capabilities,
            factory_builder=factory_builder,
        )
        return function

    return _wrap


def _register(
    name: str,
    function: Callable[..., Any],
    registry: PluginRegistry | None,
    category: PluginCategory,
    description: str,
    version: str,
    aliases: Iterable[str],
    metadata: Mapping[str, Any] | None,
    capabilities: StageCapabilities | None,
    factory_builder: Callable[[Callable[..., Any], StageCapabilities | None], Callable[..., Any]],
) -> None:
    resolved_capabilities = resolve_function_capabilities(function, capabilities, _default_capabilities(category))
    factory = factory_builder(function, resolved_capabilities)
    _attach_factory_metadata(factory, function=function, capabilities=resolved_capabilities)
    target = registry.register if registry is not None else register_user_plugin
    target(
        category,
        name,
        factory,
        description or f"Function plugin registered through @{category.value}.",
        version=version,
        aliases=aliases,
        metadata=_decorator_metadata(function, metadata),
    )


def _attach_factory_metadata(
    factory: Callable[..., Any],
    *,
    function: Callable[..., Any],
    capabilities: StageCapabilities | None,
) -> None:
    setattr(factory, "factory_path", f"{function.__module__}.{function.__qualname__}")
    if capabilities is not None:
        setattr(factory, "capabilities", capabilities)


def _decorator_metadata(function: Callable[..., Any], metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(metadata or {})
    result["source"] = "sef.decorator"
    result["function"] = f"{function.__module__}.{function.__qualname__}"
    return result


def _default_capabilities(category: PluginCategory) -> StageCapabilities:
    if category == PluginCategory.SINGLE_FRAME_PROCESSOR:
        return StageCapabilities.streaming(
            stateful=False,
            supports_frame_parallelism=True,
            realtime_safe=True,
        )
    return StageCapabilities.batch()
