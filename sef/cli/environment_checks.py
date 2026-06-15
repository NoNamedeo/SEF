from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from pathlib import Path

import sef
from sef.cli.constants import OUTPUT_DIR_NAME, PLUGIN_DIR_NAME, VIDEO_DIR_NAME
from sef.cli.diagnostics import CliDiagnostics
from sef.cli.output import print_ok


def doctor_python(diagnostics: CliDiagnostics) -> None:
    """Validate the Python runtime used by the CLI."""
    version = sys.version_info
    if version < (3, 11):
        diagnostics.add_error(
            f"Python {platform.python_version()} is not supported.",
            cause="SEF uses Python 3.11+ language/runtime features.",
            suggestion="Create a Python 3.11+ environment and reinstall SEF.",
        )
    else:
        print_ok(f"python={platform.python_version()}")


def doctor_installation() -> None:
    """Print package version and installation mode diagnostics."""
    print_ok(f"sef_version={sef_distribution_version()}")
    print_ok(f"install_path={Path(sef.__file__).resolve()}")
    print_ok(f"install_mode={installation_mode()}")


def doctor_dependencies(diagnostics: CliDiagnostics) -> None:
    """Report required core dependencies and optional feature extras."""
    required = {"numpy": "numpy", "yaml": "PyYAML"}
    optional = {
        "cv2": ("opencv", "opencv-contrib-python"),
        "matplotlib": ("visualization", "matplotlib"),
        "streamlit": ("ui", "streamlit"),
        "ultralytics": ("yolo", "ultralytics"),
        "joblib": ("pose", "joblib"),
    }
    for module_name, package_name in required.items():
        if module_importable(module_name):
            print_ok(f"core_dependency={package_name}")
        else:
            diagnostics.add_error(
                f"Missing dependency `{package_name}`.",
                cause=f"Python cannot import `{module_name}`.",
                suggestion="Install project dependencies with `pip install -e .` or `pip install -r requirements.txt`.",
            )
    for module_name, (extra_name, package_name) in optional.items():
        if module_importable(module_name):
            print_ok(f"optional_extra={extra_name} dependency={package_name}")
        else:
            diagnostics.add_warning(
                f"Optional extra `{extra_name}` is not available because `{package_name}` is not importable.",
                suggestion=f"Install it only if you use that feature: `pip install 'sef[{extra_name}]'`.",
            )


def doctor_opencv_trackers(diagnostics: CliDiagnostics) -> None:
    """Report availability of common OpenCV tracker constructors."""
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - doctor reports import diagnostics.
        diagnostics.add_warning(
            "OpenCV optional extra is not available.",
            cause=str(exc),
            suggestion="Install `sef[opencv]` if you need OpenCV-backed pipeline components.",
        )
        return

    tracker_names = ("MIL", "KCF", "CSRT", "MOSSE")
    available: list[str] = []
    for name in tracker_names:
        if hasattr(cv2, f"Tracker{name}_create") or hasattr(getattr(cv2, "legacy", object()), f"Tracker{name}_create"):
            available.append(name)
    if available:
        print_ok(f"opencv_trackers={','.join(available)}")
    else:
        diagnostics.add_warning(
            "No common OpenCV tracker constructors were detected.",
            cause="The installed OpenCV build may not include tracking APIs.",
            suggestion="Install opencv-contrib-python if you need OpenCV tracker pipelines.",
        )


def doctor_matplotlib_cache(diagnostics: CliDiagnostics) -> None:
    """Check Matplotlib cache configuration without importing Matplotlib."""
    if not module_importable("matplotlib"):
        return
    configured_cache = os.environ.get("MPLCONFIGDIR")
    if not configured_cache:
        print_ok("matplotlib_cache=not_checked")
        return
    cache_dir = Path(configured_cache)
    if cache_dir.exists() and cache_dir.is_dir() and path_writable(cache_dir):
        print_ok(f"matplotlib_cache={cache_dir}")
        return
    diagnostics.add_warning(
        f"Matplotlib cache directory is not writable: {cache_dir}",
        suggestion="Set MPLCONFIGDIR to a writable directory to avoid startup warnings.",
    )


def doctor_project_directories(diagnostics: CliDiagnostics) -> None:
    """Inspect expected project directories in the current working directory."""
    root = Path.cwd()
    for name in (PLUGIN_DIR_NAME, VIDEO_DIR_NAME, OUTPUT_DIR_NAME):
        path = root / name
        if not path.exists():
            diagnostics.add_warning(f"Directory `{name}/` does not exist.", suggestion="Run `sef init` to create it.")
            continue
        if not path.is_dir():
            diagnostics.add_error(
                f"`{name}` exists but is not a directory.",
                suggestion="Move the file or choose a clean project directory.",
            )
            continue
        if name == OUTPUT_DIR_NAME and not path_writable(path):
            diagnostics.add_error(
                f"`{name}/` is not writable.",
                suggestion="Fix permissions or pass a writable --output directory when running pipelines.",
            )
        else:
            print_ok(f"directory={name}/")


def module_importable(module_name: str) -> bool:
    """Return True when Python can resolve a module without importing it."""
    return importlib.util.find_spec(module_name) is not None


def path_writable(path: Path) -> bool:
    """Return True when the path is an existing writable directory."""
    return path.exists() and path.is_dir() and os_access(path, "w")


def os_access(path: Path, mode: str) -> bool:
    """Small wrapper around os.access for targeted tests."""
    flag = os.W_OK if mode == "w" else os.R_OK
    return os.access(path, flag)


def sef_distribution_version() -> str:
    """Return the installed SEF distribution version, or unknown in local mode."""
    try:
        return importlib.metadata.version("sef")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def installation_mode() -> str:
    """Return whether the CLI appears to run from the local checkout or a package."""
    path = Path(sef.__file__).resolve()
    cwd = Path.cwd().resolve()
    if cwd in path.parents:
        return "editable/local"
    return "package"
