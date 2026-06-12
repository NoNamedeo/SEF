"""Typed public error model for SEF integrations.

The error hierarchy gives applications stable exception families for
configuration, plugin resolution, stage execution, registry failures, and
streaming runtime failures. UI and service adapters should inspect typed fields
such as `path`, `metadata`, and `StageErrorContext` instead of parsing human
messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class SEFError(Exception):
    """Base class for public SEF library errors."""


class PipelineError(SEFError):
    """Base class for pipeline-level errors."""


class PipelineRunAlreadyActiveError(PipelineError, RuntimeError):
    """
    Raised when a pipeline id is already reserved for execution.

    The runner uses this error to reject duplicate concurrent submissions before
    they can overwrite monitor state or output-store entries.
    """


class PipelineConfigurationError(PipelineError, ValueError):
    """
    Base error for configuration-to-context failures.

    Builders should raise this family for invalid schemas, unknown plugins,
    invalid plugin parameters, or context invariant violations.

    Attributes
    ----------
    path:
        Optional dotted config path associated with the failure.
    cause:
        Original exception, when this error wraps a lower-level failure.
    metadata:
        JSON-like diagnostic metadata for UIs, APIs, and tests.
    """

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        cause: BaseException | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.cause = cause
        self.metadata = dict(metadata or {})


class ConfigSchemaError(PipelineConfigurationError):
    """Raised when a declarative pipeline config violates the supported schema."""


class ConfigVersionError(PipelineConfigurationError):
    """Raised when a declarative pipeline config uses an unsupported schema version."""

    def __init__(
        self,
        message: str,
        *,
        version: str | None = None,
        supported_versions: tuple[str, ...] = (),
        path: str | None = None,
    ) -> None:
        super().__init__(
            message,
            path=path,
            metadata={
                "version": version,
                "supported_versions": list(supported_versions),
            },
        )
        self.version = version
        self.supported_versions = supported_versions


class PipelineContextError(PipelineConfigurationError):
    """Raised when a PipelineContext violates construction invariants."""


class PluginRegistryError(SEFError):
    """Base class for plugin registry failures."""


class InvalidPluginRegistrationError(PluginRegistryError, ValueError):
    """Raised when a plugin definition is incomplete or malformed."""


class DuplicatePluginRegistrationError(InvalidPluginRegistrationError):
    """Raised when a plugin name or alias is already registered."""


class PluginResolutionError(PipelineConfigurationError):
    """Raised when a configured plugin cannot be found in the registry."""

    def __init__(
        self,
        *,
        category: str,
        name: str,
        path: str | None = None,
        available: list[str] | tuple[str, ...] = (),
        cause: BaseException | None = None,
    ) -> None:
        available_list = list(available)
        message = f"Unknown plugin '{name}'"
        if path:
            message += f" for '{path}'"
        message += f" in category '{category}'."
        if available_list:
            message += f" Available: {available_list}"
        super().__init__(
            message,
            path=path,
            cause=cause,
            metadata={
                "category": category,
                "plugin_name": name,
                "available": available_list,
            },
        )
        self.category = category
        self.name = name
        self.available = tuple(available_list)


class PluginConstructionError(PipelineConfigurationError):
    """Raised when a plugin is found but cannot be constructed."""

    def __init__(
        self,
        *,
        category: str,
        name: str,
        path: str,
        cause: BaseException,
        invalid_params: bool = False,
    ) -> None:
        prefix = "Invalid params for plugin" if invalid_params else "Failed to create plugin"
        message = f"{prefix} '{name}' at '{path}': {cause}"
        super().__init__(
            message,
            path=path,
            cause=cause,
            metadata={
                "category": category,
                "plugin_name": name,
                "invalid_params": invalid_params,
            },
        )
        self.category = category
        self.name = name
        self.invalid_params = invalid_params


class InvalidPipelineTriggerEventError(PipelineConfigurationError):
    """
    Raised when an event cannot be interpreted as a pipeline trigger.

    Branching and orchestration code use this to distinguish malformed trigger
    events from failures raised by actual pipeline execution.
    """


class LatencyPolicyError(PipelineConfigurationError):
    """Raised when stream latency-policy configuration is invalid."""


@dataclass(frozen=True, slots=True)
class StageErrorContext:
    """
    Structured metadata identifying a failed stage.

    The context is safe for monitors and UIs to serialize through `as_dict()`.
    Stage execution errors carry this object so consumers do not need to parse
    human-readable exception messages.
    """

    stage_id: str
    stage_group: str | None = None
    component_name: str | None = None
    component_type: str | None = None
    pipeline_id: str | None = None

    @classmethod
    def from_stage_id(cls, stage_id: str) -> "StageErrorContext":
        """Create a context from the stable stage id used in execution logs."""
        return cls(stage_id=stage_id, stage_group=_infer_stage_group(stage_id))

    def as_dict(self) -> dict[str, str | None]:
        """Return JSON-serializable context metadata for monitors and UIs."""
        return {
            "stage_id": self.stage_id,
            "stage_group": self.stage_group,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "pipeline_id": self.pipeline_id,
        }


class StageExecutionError(PipelineError, RuntimeError):
    """
    Raised when one pipeline stage fails.

    The original exception is preserved as ``cause`` while structured stage
    metadata is available through ``context``. This keeps UIs and runners from
    parsing human-readable messages.
    """

    def __init__(self, context: StageErrorContext | str, cause: BaseException) -> None:
        stage_context = StageErrorContext.from_stage_id(context) if isinstance(context, str) else context
        super().__init__(f"Pipeline failed at stage '{stage_context.stage_id}': {cause}")
        self.context = stage_context
        self.stage = stage_context.stage_id
        self.stage_group = stage_context.stage_group
        self.component_name = stage_context.component_name
        self.component_type = stage_context.component_type
        self.pipeline_id = stage_context.pipeline_id
        self.cause = cause


class PipelineExecutionError(StageExecutionError):
    """Backward-compatible public name for stage execution failures."""


class StreamRuntimeError(PipelineError, RuntimeError):
    """Base class for streaming runtime failures."""


class StreamAbortedError(StreamRuntimeError):
    """Raised when a stream is cooperatively aborted before normal completion."""


def _infer_stage_group(stage_id: str) -> str:
    if "." in stage_id:
        stage_id = stage_id.split(".", 1)[0]
    if "[" in stage_id:
        stage_id = stage_id.split("[", 1)[0]
    return stage_id


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
