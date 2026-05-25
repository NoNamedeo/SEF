"""Compatibility exports for the public error model.

New code should import from ``library.core.errors``. This module remains as a
stable bridge for existing pipeline imports.
"""

from library.core.errors import (
    ConfigSchemaError,
    InvalidPipelineTriggerEventError,
    LatencyPolicyError,
    PipelineConfigurationError,
    PipelineContextError,
    PipelineError,
    PipelineExecutionError,
    PipelineRunAlreadyActiveError,
    PluginConstructionError,
    PluginResolutionError,
    SEFError,
    StageErrorContext,
    StageExecutionError,
    StreamAbortedError,
    StreamRuntimeError,
)

__all__ = [
    "ConfigSchemaError",
    "InvalidPipelineTriggerEventError",
    "LatencyPolicyError",
    "PipelineConfigurationError",
    "PipelineContextError",
    "PipelineError",
    "PipelineExecutionError",
    "PipelineRunAlreadyActiveError",
    "PluginConstructionError",
    "PluginResolutionError",
    "SEFError",
    "StageErrorContext",
    "StageExecutionError",
    "StreamAbortedError",
    "StreamRuntimeError",
]
