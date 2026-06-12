from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from typing import Any

from sef.core.plugins import PluginCategory, PluginDefinition, PluginRegistry, create_builtin_registry

_USER_REGISTRY = PluginRegistry()


def register_user_plugin(
    category: str | PluginCategory,
    name: str,
    factory: Callable[..., Any],
    description: str = "",
    *,
    version: str = "1.0.0",
    aliases: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> PluginDefinition:
    """Register a process-local plugin used by the high-level ``sef`` facade."""
    return _USER_REGISTRY.register(
        category,
        name,
        factory,
        description,
        version=version,
        aliases=aliases,
        metadata=metadata,
    )


def default_registry(*, include_builtins: bool = True) -> PluginRegistry:
    """
    Return a fresh registry containing built-ins and process-local user plugins.

    The returned registry is intentionally not shared. Facades may safely add
    auto-registered classes or functions without mutating global state.
    """
    registry = _clone_registry(_builtin_registry() if include_builtins else PluginRegistry())
    _copy_definitions(_USER_REGISTRY, registry)
    return registry


def clone_registry(registry: PluginRegistry) -> PluginRegistry:
    """Return a mutable copy of an existing registry."""
    return _clone_registry(registry)


@lru_cache(maxsize=1)
def _builtin_registry() -> PluginRegistry:
    return create_builtin_registry()


def _clone_registry(source: PluginRegistry) -> PluginRegistry:
    target = PluginRegistry()
    _copy_definitions(source, target)
    return target


def _copy_definitions(source: PluginRegistry, target: PluginRegistry) -> None:
    for definition in source.list():
        target.register_definition(definition)
