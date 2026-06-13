from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


_IMPORT_HINTS = {
    "cv2": ("opencv", "opencv-contrib-python"),
    "matplotlib": ("visualization", "matplotlib"),
    "streamlit": ("ui", "streamlit"),
    "ultralytics": ("yolo", "ultralytics"),
    "joblib": ("pose", "joblib"),
    "sklearn": ("pose", "scikit-learn"),
}


class OptionalDependencyNotInstalled(ImportError):
    """Raised when a builtin adapter is used without its optional dependency group."""


@dataclass(frozen=True, slots=True)
class LazyComponentFactory:
    """
    Import a concrete builtin component only when the registry creates it.

    Builtin registration should stay cheap: listing components must not import
    OpenCV, Matplotlib, Streamlit, Ultralytics, or model-related libraries.
    """

    dotted_path: str
    extra: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        component = self.load()
        return component(*args, **kwargs)

    @property
    def factory_path(self) -> str:
        """Return the concrete component path used by diagnostics."""
        return self.dotted_path

    def load(self) -> Any:
        """Import and return the concrete component class/function."""
        module_name, _, attr_name = self.dotted_path.rpartition(".")
        if not module_name or not attr_name:
            raise ImportError(f"Invalid builtin component path: {self.dotted_path}")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise _optional_dependency_error(exc, self.extra, self.dotted_path) from exc
        try:
            return getattr(module, attr_name)
        except AttributeError as exc:
            raise ImportError(f"Builtin component '{self.dotted_path}' could not be resolved.") from exc


def lazy_component_factory(dotted_path: str, *, extra: str) -> LazyComponentFactory:
    """Return a factory that imports the concrete component on first use."""
    return LazyComponentFactory(dotted_path=dotted_path, extra=extra)


def _optional_dependency_error(error: ModuleNotFoundError, extra: str, component_path: str) -> OptionalDependencyNotInstalled:
    missing_name = str(error.name or "")
    hinted_extra, package = _IMPORT_HINTS.get(missing_name, (extra, None))
    install_target = package or missing_name or f"sef[{hinted_extra}]"
    return OptionalDependencyNotInstalled(
        f"Builtin component '{component_path}' requires optional dependency group '{hinted_extra}'. "
        f"Install it with `pip install 'sef[{hinted_extra}]'` "
        f"or install the missing package `{install_target}`."
    )
