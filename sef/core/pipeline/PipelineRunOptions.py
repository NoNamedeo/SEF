from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sef.core.pipeline.PipelineErrors import ConfigSchemaError

RUN_OPTIONS_CONFIG_KEY = "run_options"
RUN_OPTIONS_EXECUTION_PLAN_CONFIG_KEY = "execution_plan"
RUN_OPTIONS_REPRODUCIBILITY_CONFIG_KEY = "reproducibility"
_RUN_OPTIONS_CONFIG_KEYS = frozenset(
    {
        RUN_OPTIONS_EXECUTION_PLAN_CONFIG_KEY,
        RUN_OPTIONS_REPRODUCIBILITY_CONFIG_KEY,
    }
)


class PipelineExecutionPlanLevel(str, Enum):
    """Amount of execution-plan metadata attached to completed outputs."""

    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class PipelineRunOptions:
    """
    Controls optional run metadata that is not required to execute a pipeline.

    The default is intentionally lightweight: execution-plan metadata and
    reproducibility exports are omitted unless the caller explicitly requests
    them. The public ``Pipeline.execution_plan()`` method remains available
    independently from these options.
    """

    execution_plan: PipelineExecutionPlanLevel | str | bool = PipelineExecutionPlanLevel.NONE
    reproducibility: bool = False

    def __post_init__(self) -> None:
        try:
            execution_plan = _execution_plan_level_from_value(
                self.execution_plan,
                path=RUN_OPTIONS_EXECUTION_PLAN_CONFIG_KEY,
            )
        except ConfigSchemaError as exc:
            allowed = ", ".join(level.value for level in PipelineExecutionPlanLevel)
            raise ValueError(f"execution_plan must be one of: {allowed}.") from exc
        object.__setattr__(self, "execution_plan", execution_plan)

        if not isinstance(self.reproducibility, bool):
            raise TypeError("reproducibility must be a boolean.")

    @property
    def includes_execution_plan(self) -> bool:
        """Return whether the run must build an execution plan."""
        return self.execution_plan is not PipelineExecutionPlanLevel.NONE

    @classmethod
    def lightweight(cls) -> PipelineRunOptions:
        """Return options for the lowest-overhead execution path."""
        return cls()

    @classmethod
    def full(cls) -> PipelineRunOptions:
        """Return options that preserve complete plan metadata and exports."""
        return cls(
            execution_plan=PipelineExecutionPlanLevel.FULL,
            reproducibility=True,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> PipelineRunOptions:
        """
        Build run options from the optional top-level config section.

        Supported public shape:

        .. code-block:: yaml

            run_options:
              execution_plan: summary  # none | summary | full
              reproducibility: true

        A boolean ``execution_plan`` value maps to ``full`` when true and
        ``none`` when false.
        """
        if config is None:
            return cls.lightweight()
        if not isinstance(config, Mapping):
            raise ConfigSchemaError("'config' must be a mapping.", path="config")
        return cls.from_mapping(config.get(RUN_OPTIONS_CONFIG_KEY), path=RUN_OPTIONS_CONFIG_KEY)

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any] | None,
        *,
        path: str = RUN_OPTIONS_CONFIG_KEY,
    ) -> PipelineRunOptions:
        """Build run options from a validated ``run_options`` mapping."""
        if config is None:
            return cls.lightweight()
        if not isinstance(config, Mapping):
            raise ConfigSchemaError(f"'{path}' must be a mapping.", path=path)

        _validate_run_options_keys(config, path=path)
        execution_plan = _execution_plan_level_from_value(
            config.get(RUN_OPTIONS_EXECUTION_PLAN_CONFIG_KEY, PipelineExecutionPlanLevel.NONE),
            path=f"{path}.{RUN_OPTIONS_EXECUTION_PLAN_CONFIG_KEY}",
        )
        reproducibility = config.get(RUN_OPTIONS_REPRODUCIBILITY_CONFIG_KEY, False)
        if not isinstance(reproducibility, bool):
            raise ConfigSchemaError(
                f"'{path}.{RUN_OPTIONS_REPRODUCIBILITY_CONFIG_KEY}' must be a boolean.",
                path=f"{path}.{RUN_OPTIONS_REPRODUCIBILITY_CONFIG_KEY}",
            )
        return cls(execution_plan=execution_plan, reproducibility=reproducibility)

    def with_required(
        self,
        *,
        execution_plan: PipelineExecutionPlanLevel | str | bool | None = None,
        reproducibility: bool | None = None,
    ) -> PipelineRunOptions:
        """
        Return options that satisfy additional caller requirements.

        This is useful for presentation adapters such as the CLI: a config may
        request summary execution-plan metadata, while ``--explain`` requires a
        full plan. The method only upgrades requirements; it never silently
        disables config-requested metadata.
        """
        required_execution_plan = (
            _execution_plan_level_from_value(execution_plan, path=RUN_OPTIONS_EXECUTION_PLAN_CONFIG_KEY)
            if execution_plan is not None
            else self.execution_plan
        )
        return PipelineRunOptions(
            execution_plan=_max_execution_plan_level(self.execution_plan, required_execution_plan),
            reproducibility=self.reproducibility if reproducibility is None else self.reproducibility or reproducibility,
        )

    def to_config(self) -> dict[str, Any]:
        """Return the public config representation for non-default options."""
        config: dict[str, Any] = {}
        if self.execution_plan is not PipelineExecutionPlanLevel.NONE:
            config[RUN_OPTIONS_EXECUTION_PLAN_CONFIG_KEY] = self.execution_plan.value
        if self.reproducibility:
            config[RUN_OPTIONS_REPRODUCIBILITY_CONFIG_KEY] = True
        return config


_EXECUTION_PLAN_LEVEL_ORDER = {
    PipelineExecutionPlanLevel.NONE: 0,
    PipelineExecutionPlanLevel.SUMMARY: 1,
    PipelineExecutionPlanLevel.FULL: 2,
}


def _validate_run_options_keys(config: Mapping[str, Any], *, path: str) -> None:
    unknown_keys = sorted(str(key) for key in config.keys() if key not in _RUN_OPTIONS_CONFIG_KEYS)
    if unknown_keys:
        key = unknown_keys[0]
        raise ConfigSchemaError(
            f"Unsupported field '{path}.{key}'. Supported fields: execution_plan, reproducibility.",
            path=f"{path}.{key}",
        )


def _execution_plan_level_from_value(value: Any, *, path: str) -> PipelineExecutionPlanLevel:
    if isinstance(value, bool):
        return PipelineExecutionPlanLevel.FULL if value else PipelineExecutionPlanLevel.NONE
    if not isinstance(value, str):
        raise ConfigSchemaError(f"'{path}' must be one of: none, summary, full.", path=path)
    try:
        return PipelineExecutionPlanLevel(value.strip().lower())
    except ValueError as exc:
        raise ConfigSchemaError(f"'{path}' must be one of: none, summary, full.", path=path, cause=exc) from exc


def _max_execution_plan_level(
    first: PipelineExecutionPlanLevel,
    second: PipelineExecutionPlanLevel,
) -> PipelineExecutionPlanLevel:
    return first if _EXECUTION_PLAN_LEVEL_ORDER[first] >= _EXECUTION_PLAN_LEVEL_ORDER[second] else second
