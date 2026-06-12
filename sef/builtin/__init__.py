"""Built-in SEF plugins and concrete pipeline components."""

from __future__ import annotations

from pathlib import Path

from sef.core.artifacts.intermediate_frame.IntermediateFrameArtifacts import (
    set_intermediate_frame_exporter_factory,
)


def _intermediate_frame_exporter_factory(output_directory: Path):
    from sef.builtin.exporters.IntermediateFrameArtifactExporter import IntermediateFrameArtifactExporter

    return IntermediateFrameArtifactExporter(output_directory)


set_intermediate_frame_exporter_factory(_intermediate_frame_exporter_factory)


def create_builtin_registry():
    """Return a registry populated with built-in concrete components."""
    from sef.builtin.registry import create_builtin_registry as _create_builtin_registry

    return _create_builtin_registry()


__all__ = ["create_builtin_registry"]
