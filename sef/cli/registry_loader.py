from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from sef.api import default_registry
from sef.cli.constants import PLUGIN_DIR_NAME
from sef.cli.diagnostics import DiagnosticItem
from sef.core.errors import DuplicatePluginRegistrationError
from sef.core.plugins import PluginRegistry


@dataclass(frozen=True, slots=True)
class PluginImportResult:
    """Summary for local plugin loading."""

    loaded_paths: tuple[Path, ...]
    diagnostics: tuple[DiagnosticItem, ...]


class CliRegistryLoader:
    """Loads built-in SEF plugins plus local `plugins/*.py` decorator plugins."""

    def __init__(self, project_root: Path | str | None = None) -> None:
        self._project_root = Path(project_root or Path.cwd()).resolve()
        self._plugin_dir = self._project_root / PLUGIN_DIR_NAME

    def load(self) -> tuple[PluginRegistry, PluginImportResult]:
        """Import local plugins and return a fresh registry snapshot."""
        import_result = self._load_local_plugins()
        registry = default_registry(include_builtins=True)
        return registry, import_result

    def _load_local_plugins(self) -> PluginImportResult:
        loaded_paths: list[Path] = []
        diagnostics: list[DiagnosticItem] = []
        if not self._plugin_dir.exists():
            return PluginImportResult(tuple(), tuple())
        if not self._plugin_dir.is_dir():
            diagnostics.append(
                DiagnosticItem(
                    "warning",
                    f"{self._plugin_dir} exists but is not a directory.",
                    suggestion="Use a plugins/ directory containing local .py plugin modules.",
                )
            )
            return PluginImportResult(tuple(), tuple(diagnostics))

        root = str(self._project_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        for path in sorted(self._plugin_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            module_name = self._module_name(path)
            if module_name in sys.modules:
                loaded_paths.append(path)
                continue
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot create an import spec for {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                loaded_paths.append(path)
            except DuplicatePluginRegistrationError as exc:
                sys.modules.pop(module_name, None)
                diagnostics.append(
                    DiagnosticItem(
                        "warning",
                        f"Plugin module {path.name} was already registered.",
                        cause=str(exc),
                        suggestion="Use unique plugin names or restart the Python process after editing plugins.",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - local plugin import failures are diagnostics.
                sys.modules.pop(module_name, None)
                diagnostics.append(
                    DiagnosticItem(
                        "warning",
                        f"Could not import local plugin {path.name}.",
                        cause=str(exc),
                        suggestion="Fix the plugin import error. Configs that require it will fail validation.",
                    )
                )
        return PluginImportResult(tuple(loaded_paths), tuple(diagnostics))

    @staticmethod
    def _module_name(path: Path) -> str:
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
        return f"sef_local_plugins.{path.stem}_{digest}"
