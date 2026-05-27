from __future__ import annotations

from collections.abc import Callable
from typing import Any

from library.core.plugins import PluginCategory, PluginRegistry
from sef.api.function_adapters import (
    FunctionAnalyzer,
    FunctionFrameExtractor,
    FunctionFrameProcessor,
    FunctionSignalCleaner,
    FunctionSignalExtractor,
    FunctionVisualizer,
)
from sef.api.registry import register_user_plugin


def frame_extractor(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
):
    """Register a function as a frame extractor."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.FRAME_EXTRACTOR,
        factory_builder=lambda function: (lambda **params: FunctionFrameExtractor(function, params)),
    )


def signal_extractor(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
):
    """Register a function as a signal extractor."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.SIGNAL_EXTRACTOR,
        factory_builder=lambda function: (lambda **params: FunctionSignalExtractor(function, params)),
    )


def processor(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
    accepts_frame: bool = False,
):
    """Register a function as a single-frame processor."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.SINGLE_FRAME_PROCESSOR,
        factory_builder=lambda function: (
            lambda accepts_frame=accepts_frame, **params: FunctionFrameProcessor(
                function,
                params,
                accepts_frame=bool(accepts_frame),
            )
        ),
    )


def cleaner(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
):
    """Register a function as a signal cleaner."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.SIGNAL_CLEANER,
        factory_builder=lambda function: (lambda **params: FunctionSignalCleaner(function, params)),
    )


def analyzer(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
):
    """Register a function as an analyzer."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.ANALYZER,
        factory_builder=lambda function: (lambda **params: FunctionAnalyzer(function, params)),
    )


def visualizer(
    name: str | Callable[..., Any] | None = None,
    *,
    registry: PluginRegistry | None = None,
):
    """Register a function as a visualizer."""
    return _decorator(
        name,
        registry=registry,
        category=PluginCategory.VISUALIZER,
        factory_builder=lambda function: (lambda **params: FunctionVisualizer(function, params)),
    )


def _decorator(
    name: str | Callable[..., Any] | None,
    *,
    registry: PluginRegistry | None,
    category: PluginCategory,
    factory_builder: Callable[[Callable[..., Any]], Callable[..., Any]],
):
    if callable(name) and not isinstance(name, str):
        function = name
        _register(function.__name__, function, registry, category, factory_builder)
        return function

    def _wrap(function: Callable[..., Any]):
        plugin_name = name or function.__name__
        _register(str(plugin_name), function, registry, category, factory_builder)
        return function

    return _wrap


def _register(
    name: str,
    function: Callable[..., Any],
    registry: PluginRegistry | None,
    category: PluginCategory,
    factory_builder: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> None:
    target = registry.register if registry is not None else register_user_plugin
    target(
        category,
        name,
        factory_builder(function),
        f"Function plugin registered through @{category.value}.",
        metadata={"source": "sef.decorator", "function": f"{function.__module__}.{function.__qualname__}"},
    )
