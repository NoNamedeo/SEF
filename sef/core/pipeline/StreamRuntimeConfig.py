from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from sef.core.pipeline.LatencyPolicy import LatencyPolicyConfig
from sef.core.pipeline.PipelineErrors import ConfigSchemaError


@dataclass(frozen=True, slots=True)
class StreamRuntimeConfig:
    """
    Bounded-buffer and latency settings for adaptive streaming execution.

    The config controls queue capacities between streaming stages and selects
    the frame latency policy used by streaming frame extractors. It is immutable
    and safe to store in `PipelineContext` or reproducibility metadata.

    Attributes
    ----------
    frame_buffer_size:
        Public frame queue capacity between frame source and downstream stages.
    signal_buffer_size:
        Public signal queue capacity between signal stages and analyzers.
    data_buffer_size:
        Public data queue capacity between streaming analyzers and visualizers.
    latency_policy:
        Serializable latency-policy selector.
    """

    frame_buffer_size: int = 8
    signal_buffer_size: int = 8
    data_buffer_size: int = 8
    latency_policy: LatencyPolicyConfig = field(default_factory=LatencyPolicyConfig)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        path: str = "run.runtime",
    ) -> StreamRuntimeConfig:
        """
        Parse runtime settings from a declarative config mapping.

        Parameters
        ----------
        value:
            Mapping from `run.runtime`, or `None` for defaults.

        Returns
        -------
        StreamRuntimeConfig
            Validated runtime config.

        Raises
        ------
        ConfigSchemaError
            If `run.runtime` or a buffer size has an invalid shape.
        LatencyPolicyError
            If the nested latency-policy config is invalid.
        """
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ConfigSchemaError(f"{path} must be a mapping.", path=path)
        frame_buffer_size = cls._positive_int(value.get("frame_buffer_size", 8), f"{path}.frame_buffer_size")
        signal_buffer_size = cls._positive_int(
            value.get("signal_buffer_size", frame_buffer_size),
            f"{path}.signal_buffer_size",
        )
        data_buffer_size = cls._positive_int(
            value.get("data_buffer_size", signal_buffer_size),
            f"{path}.data_buffer_size",
        )
        config = cls(
            frame_buffer_size=frame_buffer_size,
            signal_buffer_size=signal_buffer_size,
            data_buffer_size=data_buffer_size,
            latency_policy=LatencyPolicyConfig.from_mapping(value.get("latency_policy"), path=f"{path}.latency_policy"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """
        Validate buffer-size invariants.

        Raises
        ------
        ConfigSchemaError
            If any public buffer capacity is less than one.
        """
        if self.frame_buffer_size <= 0:
            raise ConfigSchemaError(
                "runtime.frame_buffer_size must be greater than 0.",
                path="run.runtime.frame_buffer_size",
            )
        if self.signal_buffer_size <= 0:
            raise ConfigSchemaError(
                "runtime.signal_buffer_size must be greater than 0.",
                path="run.runtime.signal_buffer_size",
            )
        if self.data_buffer_size <= 0:
            raise ConfigSchemaError(
                "runtime.data_buffer_size must be greater than 0.",
                path="run.runtime.data_buffer_size",
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for exported configs and plans."""
        return {
            "frame_buffer_size": self.frame_buffer_size,
            "signal_buffer_size": self.signal_buffer_size,
            "data_buffer_size": self.data_buffer_size,
            "latency_policy": self.latency_policy.as_dict(),
        }

    @staticmethod
    def _positive_int(value: Any, path: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigSchemaError(f"{path} must be an integer greater than 0.", path=path, cause=exc) from exc
        if parsed <= 0:
            raise ConfigSchemaError(f"{path} must be greater than 0.", path=path)
        return parsed
