from __future__ import annotations

import pytest

from sef.core import (
    CURRENT_PIPELINE_CONFIG_VERSION,
    ConfigPipelineBuilder,
    ConfigSchemaError,
    ConfigVersionError,
    Frame,
    FrameBuffer,
    PipelineConfigurationError,
    PipelineContext,
    PipelineContextError,
    PipelineExecutionError,
    PipelineExecutionPlanLevel,
    PipelineRunOptions,
    PluginCategory,
    PluginResolutionError,
    SEFError,
    StageErrorContext,
    StageExecutionError,
    TextArtifact,
    VisualArtifact,
)
from sef.core.artifacts.buffer import SignalBuffer
from sef.core.artifacts.data import COCOPoseTennisFrameData
from sef.core.errors import LatencyPolicyError
from sef.core.pipeline import LatencyPolicyConfig, StreamRuntimeConfig
from sef.core.plugins import PluginRegistry
from sef.core.visualization import ImageArtifact, PipelineOutputs


def test_core_public_api_exports_stable_entry_points() -> None:
    assert ConfigPipelineBuilder is not None
    assert Frame is not None
    assert FrameBuffer is not None
    assert PipelineContext is not None
    assert PluginCategory.FRAME_EXTRACTOR == "frame_extractor"
    assert TextArtifact is not None
    assert VisualArtifact is not None
    assert COCOPoseTennisFrameData is not None
    assert SignalBuffer is not None
    assert ImageArtifact is not None
    assert PipelineOutputs is not None
    assert PipelineRunOptions.lightweight().execution_plan is PipelineExecutionPlanLevel.NONE


def test_pipeline_error_exports_are_identity_stable() -> None:
    from sef import PipelineExecutionError as TopLevelPipelineExecutionError
    from sef.core.pipeline import PipelineExecutionError as PipelinePackageExecutionError

    assert PipelinePackageExecutionError is PipelineExecutionError
    assert TopLevelPipelineExecutionError is PipelineExecutionError
    assert issubclass(PipelineExecutionError, StageExecutionError)
    assert issubclass(PipelineConfigurationError, SEFError)
    assert issubclass(ConfigVersionError, PipelineConfigurationError)
    assert CURRENT_PIPELINE_CONFIG_VERSION == "1.0"


def test_unknown_plugin_raises_typed_resolution_error() -> None:
    builder = ConfigPipelineBuilder(PluginRegistry())
    config = {
        "pipeline": {
            "frame_extractor": {"name": "missing"},
            "signal_extractor": {"name": "unused"},
            "analyzers": [{"name": "unused"}],
        }
    }

    with pytest.raises(PluginResolutionError) as raised:
        builder.build_context(config)

    error = raised.value
    assert isinstance(error, PipelineConfigurationError)
    assert isinstance(error, ValueError)
    assert error.path == "pipeline.frame_extractor"
    assert error.category == "frame_extractor"
    assert error.name == "missing"
    assert error.available == ()


def test_runtime_config_raises_schema_errors_with_paths() -> None:
    with pytest.raises(ConfigSchemaError) as raised:
        StreamRuntimeConfig.from_mapping({"frame_buffer_size": 0})

    assert raised.value.path == "pipeline.runtime.frame_buffer_size"


def test_latency_policy_raises_typed_policy_errors() -> None:
    with pytest.raises(LatencyPolicyError) as raised:
        LatencyPolicyConfig(name="unsupported").create()

    assert raised.value.path == "pipeline.runtime.latency_policy.name"

    with pytest.raises(LatencyPolicyError) as bad_param:
        LatencyPolicyConfig(name="adaptive_sampling", params={"min_interval": "bad"}).create()

    assert bad_param.value.path == "pipeline.runtime.latency_policy.params.min_interval"


def test_pipeline_context_errors_remain_value_errors() -> None:
    with pytest.raises(PipelineContextError) as raised:
        PipelineContext(frame_extractor=None, signal_extractor=object(), analyzers=[object()])

    assert isinstance(raised.value, PipelineConfigurationError)
    assert isinstance(raised.value, ValueError)
    assert raised.value.path == "frame_extractor"


def test_stage_execution_error_preserves_structured_context_and_compatibility() -> None:
    cause = RuntimeError("boom")
    context = StageErrorContext(stage_id="analysis[0]", component_name="pose")

    error = PipelineExecutionError(context, cause)

    assert isinstance(error, StageExecutionError)
    assert isinstance(error, RuntimeError)
    assert error.stage == "analysis[0]"
    assert error.stage_group is None
    assert error.component_name == "pose"
    assert error.cause is cause
