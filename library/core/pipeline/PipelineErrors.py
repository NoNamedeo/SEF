from __future__ import annotations


class PipelineRunAlreadyActiveError(RuntimeError):
    """Raised when a pipeline id is already reserved for execution."""


class InvalidPipelineTriggerEventError(ValueError):
    """Raised when an Event cannot be interpreted as a pipeline trigger."""


class PipelineConfigurationError(ValueError):
    """Raised when a pipeline configuration cannot be converted to a context."""
