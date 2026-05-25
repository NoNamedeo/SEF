from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from library.core.pipeline.LatencyPolicy import LatencyPolicyConfig
from library.core.pipeline.PipelineErrors import ConfigSchemaError


@dataclass(frozen=True, slots=True)
class StreamRuntimeConfig:
    """Bounded-buffer and latency settings used by the adaptive pipeline runtime."""

    frame_buffer_size: int = 8
    signal_buffer_size: int = 8
    data_buffer_size: int = 8
    latency_policy: LatencyPolicyConfig = field(default_factory=LatencyPolicyConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> StreamRuntimeConfig:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ConfigSchemaError("pipeline.runtime must be a mapping.", path="pipeline.runtime")
        frame_buffer_size = cls._positive_int(value.get("frame_buffer_size", 8), "pipeline.runtime.frame_buffer_size")
        signal_buffer_size = cls._positive_int(
            value.get("signal_buffer_size", frame_buffer_size),
            "pipeline.runtime.signal_buffer_size",
        )
        data_buffer_size = cls._positive_int(
            value.get("data_buffer_size", signal_buffer_size),
            "pipeline.runtime.data_buffer_size",
        )
        config = cls(
            frame_buffer_size=frame_buffer_size,
            signal_buffer_size=signal_buffer_size,
            data_buffer_size=data_buffer_size,
            latency_policy=LatencyPolicyConfig.from_mapping(value.get("latency_policy")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.frame_buffer_size <= 0:
            raise ConfigSchemaError(
                "runtime.frame_buffer_size must be greater than 0.",
                path="pipeline.runtime.frame_buffer_size",
            )
        if self.signal_buffer_size <= 0:
            raise ConfigSchemaError(
                "runtime.signal_buffer_size must be greater than 0.",
                path="pipeline.runtime.signal_buffer_size",
            )
        if self.data_buffer_size <= 0:
            raise ConfigSchemaError(
                "runtime.data_buffer_size must be greater than 0.",
                path="pipeline.runtime.data_buffer_size",
            )

    def as_dict(self) -> dict[str, Any]:
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
