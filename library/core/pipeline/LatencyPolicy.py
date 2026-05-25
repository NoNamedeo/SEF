from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from library.core.artifacts.Frame import Frame
from library.core.interfaces.BufferContracts import IFrameBuffer
from library.core.pipeline.PipelineErrors import LatencyPolicyError


@dataclass(frozen=True, slots=True)
class LatencyPolicyConfig:
    """Serializable runtime configuration for frame-stream latency handling."""

    name: str = "blocking"
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> LatencyPolicyConfig:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise LatencyPolicyError("latency_policy must be a mapping.", path="pipeline.runtime.latency_policy")
        name = str(value.get("name", "blocking")).strip().lower()
        if not name:
            raise LatencyPolicyError("latency_policy.name cannot be empty.", path="pipeline.runtime.latency_policy.name")
        params = value.get("params", {})
        if not isinstance(params, Mapping):
            raise LatencyPolicyError(
                "latency_policy.params must be a mapping.",
                path="pipeline.runtime.latency_policy.params",
            )
        return cls(name=name, params=dict(params))

    def create(self) -> FrameLatencyPolicy:
        """Create a fresh stateful policy instance for one pipeline run."""
        if self.name == "blocking":
            return BlockingFrameLatencyPolicy()
        if self.name == "drop_newest":
            return DropNewestFrameLatencyPolicy()
        if self.name == "drop_oldest":
            return DropOldestFrameLatencyPolicy()
        if self.name == "adaptive_sampling":
            return AdaptiveSamplingFrameLatencyPolicy.from_mapping(self.params)
        raise LatencyPolicyError(
            "Unsupported latency policy "
            f"'{self.name}'. Supported values: blocking, drop_newest, drop_oldest, adaptive_sampling.",
            path="pipeline.runtime.latency_policy.name",
        )

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


class FrameLatencyPolicy(ABC):
    """Strategy used by live/streaming extractors when the frame queue is under pressure."""

    @abstractmethod
    def publish(self, frame: Frame, output_buffer: IFrameBuffer) -> bool:
        """Publish ``frame`` to ``output_buffer`` or drop it. Return True when accepted."""

    def metrics(self) -> dict[str, Any]:
        """Return runtime counters for observability."""
        return {}


class BlockingFrameLatencyPolicy(FrameLatencyPolicy):
    """Preserve every frame and let upstream block when the bounded queue is full."""

    def __init__(self) -> None:
        self.accepted_frames = 0

    def publish(self, frame: Frame, output_buffer: IFrameBuffer) -> bool:
        output_buffer.put(frame)
        self.accepted_frames += 1
        return True

    def metrics(self) -> dict[str, Any]:
        return {"accepted_frames": self.accepted_frames, "dropped_frames": 0}


class DropNewestFrameLatencyPolicy(FrameLatencyPolicy):
    """Drop the incoming frame when the downstream queue is full."""

    def __init__(self) -> None:
        self.accepted_frames = 0
        self.dropped_frames = 0

    def publish(self, frame: Frame, output_buffer: IFrameBuffer) -> bool:
        if output_buffer.try_put(frame):
            self.accepted_frames += 1
            return True
        self.dropped_frames += 1
        return False

    def metrics(self) -> dict[str, Any]:
        return {"accepted_frames": self.accepted_frames, "dropped_frames": self.dropped_frames}


class DropOldestFrameLatencyPolicy(FrameLatencyPolicy):
    """Keep latency low by discarding the oldest queued frame when needed."""

    def __init__(self) -> None:
        self.accepted_frames = 0
        self.dropped_frames = 0

    def publish(self, frame: Frame, output_buffer: IFrameBuffer) -> bool:
        if output_buffer.try_put(frame):
            self.accepted_frames += 1
            return True
        if output_buffer.drop_oldest() is not None:
            self.dropped_frames += 1
        if output_buffer.try_put(frame):
            self.accepted_frames += 1
            return True
        self.dropped_frames += 1
        return False

    def metrics(self) -> dict[str, Any]:
        return {"accepted_frames": self.accepted_frames, "dropped_frames": self.dropped_frames}


class AdaptiveSamplingFrameLatencyPolicy(FrameLatencyPolicy):
    """
    Reduce processed FPS by increasing the sampling interval under queue pressure.

    The policy is intentionally simple and deterministic: it observes the target
    queue fill ratio before each publish and adjusts the sampling interval within
    configured bounds. This stabilizes live preview latency without adding worker
    coordination complexity.
    """

    def __init__(
        self,
        *,
        min_interval: int = 1,
        max_interval: int = 8,
        high_watermark: float = 0.75,
        low_watermark: float = 0.25,
    ) -> None:
        if min_interval <= 0:
            raise LatencyPolicyError(
                "min_interval must be greater than 0.",
                path="pipeline.runtime.latency_policy.params.min_interval",
            )
        if max_interval < min_interval:
            raise LatencyPolicyError(
                "max_interval must be greater than or equal to min_interval.",
                path="pipeline.runtime.latency_policy.params.max_interval",
            )
        if not (0.0 <= low_watermark <= high_watermark <= 1.0):
            raise LatencyPolicyError(
                "watermarks must satisfy 0 <= low_watermark <= high_watermark <= 1.",
                path="pipeline.runtime.latency_policy.params",
            )
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        self.high_watermark = float(high_watermark)
        self.low_watermark = float(low_watermark)
        self.current_interval = self.min_interval
        self.seen_frames = 0
        self.accepted_frames = 0
        self.dropped_frames = 0

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> AdaptiveSamplingFrameLatencyPolicy:
        return cls(
            min_interval=_coerce_int(
                params.get("min_interval", 1),
                "pipeline.runtime.latency_policy.params.min_interval",
            ),
            max_interval=_coerce_int(
                params.get("max_interval", 8),
                "pipeline.runtime.latency_policy.params.max_interval",
            ),
            high_watermark=_coerce_float(
                params.get("high_watermark", 0.75),
                "pipeline.runtime.latency_policy.params.high_watermark",
            ),
            low_watermark=_coerce_float(
                params.get("low_watermark", 0.25),
                "pipeline.runtime.latency_policy.params.low_watermark",
            ),
        )

    def publish(self, frame: Frame, output_buffer: IFrameBuffer) -> bool:
        self.seen_frames += 1
        self._update_interval(output_buffer)
        if (self.seen_frames - 1) % self.current_interval != 0:
            self.dropped_frames += 1
            return False
        if output_buffer.try_put(frame):
            self.accepted_frames += 1
            return True
        if output_buffer.drop_oldest() is not None:
            self.dropped_frames += 1
        if output_buffer.try_put(frame):
            self.accepted_frames += 1
            return True
        self.dropped_frames += 1
        return False

    def metrics(self) -> dict[str, Any]:
        return {
            "seen_frames": self.seen_frames,
            "accepted_frames": self.accepted_frames,
            "dropped_frames": self.dropped_frames,
            "current_interval": self.current_interval,
        }

    def _update_interval(self, output_buffer: IFrameBuffer) -> None:
        fill_ratio = output_buffer.fill_ratio()
        if fill_ratio >= self.high_watermark:
            self.current_interval = min(self.max_interval, self.current_interval + 1)
        elif fill_ratio <= self.low_watermark:
            self.current_interval = max(self.min_interval, self.current_interval - 1)


def _coerce_int(value: Any, path: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise LatencyPolicyError(f"{path} must be an integer.", path=path, cause=exc) from exc


def _coerce_float(value: Any, path: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise LatencyPolicyError(f"{path} must be a number.", path=path, cause=exc) from exc
