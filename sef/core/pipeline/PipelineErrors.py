"""Compatibility exports for the public error model.

New code should import from ``sef.core.errors``. This module remains as a
stable bridge for existing pipeline imports.
"""

from sef.core.errors import (
    ConfigSchemaError,
    ConfigVersionError,
    DuplicatePluginRegistrationError,
    InvalidPipelineTriggerEventError,
    InvalidPluginRegistrationError,
    LatencyPolicyError,
    PipelineConfigurationError,
    PipelineContextError,
    PipelineError,
    PipelineExecutionError,
    PipelineRunAlreadyActiveError,
    PluginConstructionError,
    PluginRegistryError,
    PluginResolutionError,
    SEFError,
    StageErrorContext,
    StageExecutionError,
    StreamAbortedError,
    StreamRuntimeError,
)

__all__ = [
    "ConfigSchemaError",
    "ConfigVersionError",
    "DuplicatePluginRegistrationError",
    "InvalidPipelineTriggerEventError",
    "InvalidPluginRegistrationError",
    "LatencyPolicyError",
    "PipelineConfigurationError",
    "PipelineContextError",
    "PipelineError",
    "PipelineExecutionError",
    "PipelineRunAlreadyActiveError",
    "PluginConstructionError",
    "PluginRegistryError",
    "PluginResolutionError",
    "SEFError",
    "StageErrorContext",
    "StageExecutionError",
    "StreamAbortedError",
    "StreamRuntimeError",
]
