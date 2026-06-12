"""Public plugin registry contracts.

The registry package exposes category identifiers, immutable plugin
descriptors, the production registry implementation, and the built-in registry
factory. Declarative builders depend on this layer to resolve configuration
entries without importing concrete OpenCV, YOLO, analysis, or visualization
implementations directly.
"""

from library.core.plugins.PluginRegistry import (
    PluginCategory,
    PluginDefinition,
    PluginRegistry,
    create_builtin_registry,
)

__all__ = [
    "PluginCategory",
    "PluginDefinition",
    "PluginRegistry",
    "create_builtin_registry",
]
