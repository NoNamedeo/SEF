from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any


def dotted_path(value: Any) -> str:
    """Return an import-oriented dotted path for classes, callables, and objects."""
    target = value if isinstance(value, type) else type(value)
    module = getattr(target, "__module__", "")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", type(target).__name__))
    return f"{module}.{qualname}" if module else qualname


def to_exportable_data(value: Any) -> Any:
    """
    Convert common Python objects into deterministic JSON/YAML-safe values.

    Export metadata should stay shareable even when artifacts contain paths,
    datetimes, tuples, enums, or NumPy scalars. Unknown objects are represented
    by type and repr so the export remains inspectable without pretending the
    value can always be passed back into a constructor.
    """
    scalar = _numpy_scalar(value)
    if scalar is not _MISSING:
        return to_exportable_data(scalar)

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_exportable_data(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return to_exportable_data(asdict(value))
    if isinstance(value, bytes | bytearray | memoryview):
        return {"type": "bytes", "size_bytes": len(value)}
    if isinstance(value, set | frozenset):
        return [to_exportable_data(item) for item in sorted(value, key=repr)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray | memoryview):
        return [to_exportable_data(item) for item in value]
    if hasattr(value, "__fspath__"):
        return str(Path(value))
    if callable(value):
        return {"type": "callable", "path": _callable_path(value)}
    return {"type": dotted_path(value), "repr": repr(value)}


def is_rebuildable_param(value: Any) -> bool:
    """Return True when a value can safely be emitted as constructor config."""
    scalar = _numpy_scalar(value)
    if scalar is not _MISSING:
        return is_rebuildable_param(scalar)

    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, Path | datetime | date | Enum):
        return True
    if isinstance(value, Mapping):
        return all(isinstance(key, str | int | float | bool) and is_rebuildable_param(item) for key, item in value.items())
    if is_dataclass(value) and not isinstance(value, type):
        return is_rebuildable_param(asdict(value))
    if isinstance(value, set | frozenset):
        return all(is_rebuildable_param(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray | memoryview):
        return all(is_rebuildable_param(item) for item in value)
    if hasattr(value, "__fspath__"):
        return True
    return False


def json_dumps(value: Mapping[str, Any]) -> str:
    """Serialize export data as stable, human-readable JSON."""
    return json.dumps(to_exportable_data(value), indent=2, sort_keys=False)


def yaml_dumps(value: Mapping[str, Any]) -> str:
    """Serialize export data as simple YAML without requiring a runtime dependency."""
    exportable = to_exportable_data(value)
    lines = _yaml_lines(exportable, indent=0)
    return "\n".join(lines).rstrip() + "\n"


def _yaml_lines(value: Any, indent: int) -> list[str]:
    prefix = "  " * indent
    if isinstance(value, Mapping):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            rendered_key = _yaml_key(str(key))
            if _is_yaml_scalar(item):
                lines.append(f"{prefix}{rendered_key}: {_yaml_scalar(item)}")
            else:
                lines.append(f"{prefix}{rendered_key}:")
                lines.extend(_yaml_lines(item, indent + 1))
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if _is_yaml_scalar(item):
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
            else:
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 1))
        return lines

    return [f"{prefix}{_yaml_scalar(value)}"]


def _yaml_key(value: str) -> str:
    if value.replace("_", "").replace("-", "").isalnum():
        return value
    return json.dumps(value)


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(str(value))


def _is_yaml_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _callable_path(value: Any) -> str:
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", repr(value)))
    return f"{module}.{qualname}" if module else qualname


class _Missing:
    pass


_MISSING = _Missing()


def _numpy_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if item is None or not callable(item):
        return _MISSING
    try:
        scalar = item()
    except Exception:
        return _MISSING
    if scalar is value:
        return _MISSING
    return scalar
