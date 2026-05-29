from __future__ import annotations

import numpy as np
import pytest

from library.core.artifacts.Frame import Frame
from library.core.artifacts.buffer.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.data.TwoDimGraphData import TwoDimGraphData
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.PipelineConfigExporter import PipelineConfigExporter
from library.core.pipeline.PipelineConfigVersioning import (
    CURRENT_PIPELINE_CONFIG_VERSION,
    PipelineConfigMigration,
    PipelineConfigVersionManager,
)
from library.core.pipeline.PipelineErrors import ConfigVersionError
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry


class VersionedFrameExtractor(IFrameExtractor):
    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(1)
        buffer.put(Frame(image=np.zeros((2, 2, 3), dtype=np.uint8), index=0))
        buffer.close()
        return buffer


class VersionedSignalExtractor(ISignalExtractor):
    def extract(self, buffer: FrameBuffer) -> ISignal:
        return Signal([])


class VersionedAnalyzer(IAnalyzer):
    def analyze(self, signal: ISignal) -> IData:
        return TwoDimGraphData(x=[], y=[], title="versioned")


def test_builder_accepts_explicit_current_schema_version() -> None:
    context = ConfigPipelineBuilder(_registry()).build_context(
        {
            "schema_version": CURRENT_PIPELINE_CONFIG_VERSION,
            "pipeline": _pipeline_section(),
        }
    )

    assert context.source_config["schema_version"] == CURRENT_PIPELINE_CONFIG_VERSION
    assert context.source_config["pipeline"]["frame_extractor"]["name"] == "versioned_frames"


def test_builder_normalizes_legacy_unversioned_configs() -> None:
    context = ConfigPipelineBuilder(_registry()).build_context({"pipeline": _pipeline_section()})

    assert context.source_config["schema_version"] == CURRENT_PIPELINE_CONFIG_VERSION


def test_builder_rejects_unsupported_schema_version() -> None:
    builder = ConfigPipelineBuilder(_registry())

    with pytest.raises(ConfigVersionError) as raised:
        builder.build_context({"schema_version": "99.0", "pipeline": _pipeline_section()})

    assert raised.value.path == "schema_version"
    assert raised.value.version == "99.0"
    assert raised.value.supported_versions == (CURRENT_PIPELINE_CONFIG_VERSION,)


def test_config_version_manager_supports_explicit_migration_steps() -> None:
    def migrate_legacy(config):
        migrated = dict(config)
        migrated["pipeline"] = migrated.pop("legacy_pipeline")
        return migrated

    manager = PipelineConfigVersionManager(
        migrations=(
            PipelineConfigMigration(
                source_version="0.9",
                target_version=CURRENT_PIPELINE_CONFIG_VERSION,
                migrate=migrate_legacy,
            ),
        )
    )

    versioned_config = manager.normalize(
        {
            "schema_version": "0.9",
            "legacy_pipeline": _pipeline_section(),
        }
    )

    assert versioned_config.schema_version == CURRENT_PIPELINE_CONFIG_VERSION
    assert versioned_config.applied_migrations == ("0.9->1.0",)
    assert versioned_config.pipeline["analyzers"][0]["name"] == "versioned_analyzer"


def test_exporter_emits_current_schema_version() -> None:
    context = ConfigPipelineBuilder(_registry()).build_context({"pipeline": _pipeline_section()})

    exported = PipelineConfigExporter(_registry()).export(context)

    assert exported["schema_version"] == CURRENT_PIPELINE_CONFIG_VERSION


def _registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(PluginCategory.FRAME_EXTRACTOR, "versioned_frames", VersionedFrameExtractor)
    registry.register(PluginCategory.SIGNAL_EXTRACTOR, "versioned_signals", VersionedSignalExtractor)
    registry.register(PluginCategory.ANALYZER, "versioned_analyzer", VersionedAnalyzer)
    return registry


def _pipeline_section() -> dict:
    return {
        "frame_extractor": {"name": "versioned_frames"},
        "signal_extractor": {"name": "versioned_signals"},
        "analyzers": [{"name": "versioned_analyzer"}],
    }
