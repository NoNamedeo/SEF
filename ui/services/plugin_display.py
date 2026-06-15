"""Presentation helpers for rendering plugin registry descriptors."""

from __future__ import annotations

import json
from typing import Any


def plugin_factory_label(plugin: Any) -> str:
    """Return a stable, UI-safe label for class factories and lazy factories."""
    explicit_path = getattr(plugin, "factory_path", None)
    if isinstance(explicit_path, str) and explicit_path:
        return explicit_path.rsplit(".", 1)[-1]
    factory = getattr(plugin, "factory", plugin)
    factory_path = getattr(factory, "factory_path", None)
    if isinstance(factory_path, str) and factory_path:
        return factory_path.rsplit(".", 1)[-1]
    return getattr(factory, "__name__", type(factory).__name__)


def plugin_capabilities_label(plugin: Any) -> str:
    """Return a compact capability summary without forcing lazy imports."""
    capabilities = getattr(plugin.factory, "capabilities", None)
    if capabilities is None:
        return "-"
    as_dict = getattr(capabilities, "as_dict", None)
    if callable(as_dict):
        return json.dumps(as_dict(), sort_keys=True)
    return str(capabilities)
