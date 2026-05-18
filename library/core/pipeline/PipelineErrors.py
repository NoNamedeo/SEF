from __future__ import annotations


class PipelineRunAlreadyActiveError(RuntimeError):
    """
    Raised when a pipeline id is already reserved for execution.

    The runner uses this error to reject duplicate concurrent submissions before
    they can overwrite monitor state or output-store entries.
    """


class InvalidPipelineTriggerEventError(ValueError):
    """
    Raised when an Event cannot be interpreted as a pipeline trigger.

    Branching and orchestration code use this to distinguish malformed trigger
    events from failures raised by actual pipeline execution.
    """


class PipelineConfigurationError(ValueError):
    """
    Raised when a pipeline configuration cannot be converted to a context.

    Builders should raise this error for invalid schemas, unknown plugin names,
    invalid plugin parameters, or context invariant violations.
    """


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
