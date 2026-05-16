from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from library.core.pipeline.LatencyPolicy import LatencyPolicyConfig


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
            raise ValueError("pipeline.runtime must be a mapping.")
        frame_buffer_size = int(value.get("frame_buffer_size", 8))
        signal_buffer_size = int(value.get("signal_buffer_size", frame_buffer_size))
        data_buffer_size = int(value.get("data_buffer_size", signal_buffer_size))
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
            raise ValueError("runtime.frame_buffer_size must be greater than 0.")
        if self.signal_buffer_size <= 0:
            raise ValueError("runtime.signal_buffer_size must be greater than 0.")
        if self.data_buffer_size <= 0:
            raise ValueError("runtime.data_buffer_size must be greater than 0.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_buffer_size": self.frame_buffer_size,
            "signal_buffer_size": self.signal_buffer_size,
            "data_buffer_size": self.data_buffer_size,
            "latency_policy": self.latency_policy.as_dict(),
        }
