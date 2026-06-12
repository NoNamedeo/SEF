"""Public plugin registry contracts.

The registry package exposes category identifiers, immutable plugin
descriptors, and the production registry implementation. Declarative builders
depend on this layer to resolve configuration entries without importing
concrete OpenCV, YOLO, analysis, or visualization implementations directly.
"""

from sef.core.plugins.PluginRegistry import (
    PluginCategory,
    PluginDefinition,
    PluginRegistry,
)

__all__ = [
    "PluginCategory",
    "PluginDefinition",
    "PluginRegistry",
]
