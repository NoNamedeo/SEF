"""Helpers for package-level lazy exports."""

from __future__ import annotations

import sys
from types import ModuleType


class _LazyExportModule(ModuleType):
    def __getattribute__(self, name: str):
        namespace = ModuleType.__getattribute__(self, "__dict__")
        exports = namespace.get("_EXPORTS", {})

        try:
            value = ModuleType.__getattribute__(self, name)
        except AttributeError:
            resolver = namespace.get("__getattr__")
            if resolver is not None:
                return resolver(name)
            raise

        if name in exports and isinstance(value, ModuleType):
            module_name, _attr_name = exports[name]
            if value.__name__ == module_name:
                return namespace["__getattr__"](name)

        return value


def install_lazy_exports(module_name: str) -> None:
    sys.modules[module_name].__class__ = _LazyExportModule
