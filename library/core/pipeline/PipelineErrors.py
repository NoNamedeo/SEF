from __future__ import annotations


class PipelineRunAlreadyActiveError(RuntimeError):
    """Raised when a pipeline id is already reserved for execution."""


class InvalidPipelineTriggerEventError(ValueError):
    """Raised when an Event cannot be interpreted as a pipeline trigger."""


class PipelineConfigurationError(ValueError):
    """Raised when a pipeline configuration cannot be converted to a context."""


class PipelineExecutionError(RuntimeError):
    """
    Raised when a single pipeline stage fails.

    The public error keeps the original cause available while adding the stage
    name that failed. Runners and UIs can then report precise failures without
    parsing log messages.
    """

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"Pipeline failed at stage '{stage}': {cause}")
        self.stage = stage
        self.cause = cause
