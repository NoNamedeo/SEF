from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from sef.core.artifacts.Frame import Frame
from sef.core.interfaces.BufferContracts import IFrameBuffer
from sef.core.pipeline.PipelineErrors import LatencyPolicyError


@dataclass(frozen=True, slots=True)
class LatencyPolicyConfig:
    """
    Serializable selector for frame-stream latency behavior.

    The config is stored in pipeline configs and run metadata. Calling
    `create()` returns a fresh policy instance because concrete policies keep
    per-run counters and, for adaptive sampling, per-run control state.

    Attributes
    ----------
    name:
        One of `blocking`, `drop_newest`, `drop_oldest`, or
        `adaptive_sampling`.
    params:
        Policy-specific parameters. Only adaptive sampling currently consumes
        parameters.
    """

    name: str = "blocking"
    params: Mapping[str, Any] = field(default_factory=dict)
    _path: str = field(default="run.runtime.latency_policy", repr=False, compare=False)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        path: str = "run.runtime.latency_policy",
    ) -> LatencyPolicyConfig:
        """
        Parse a latency policy mapping from declarative config.

        Parameters
        ----------
        value:
            Mapping from `run.runtime.latency_policy`, or `None` to select
            the default blocking policy.

        Returns
        -------
        LatencyPolicyConfig
            Normalized, lower-case policy selector.

        Raises
        ------
        LatencyPolicyError
            If the mapping shape, policy name, or params section is invalid.
        """
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise LatencyPolicyError("latency_policy must be a mapping.", path=path)
        name = str(value.get("name", "blocking")).strip().lower()
        if not name:
            raise LatencyPolicyError("latency_policy.name cannot be empty.", path=f"{path}.name")
        params = value.get("params", {})
        if not isinstance(params, Mapping):
            raise LatencyPolicyError(
                "latency_policy.params must be a mapping.",
                path=f"{path}.params",
            )
        return cls(name=name, params=dict(params), _path=path)

    def create(self) -> FrameLatencyPolicy:
        """
        Create a fresh stateful policy instance for one pipeline run.

        Returns
        -------
        FrameLatencyPolicy
            Runtime strategy used by streaming frame extractors.

        Raises
        ------
        LatencyPolicyError
            If `name` is not supported or policy-specific params are invalid.
        """
        if self.name == "blocking":
            return BlockingFrameLatencyPolicy()
        if self.name == "drop_newest":
            return DropNewestFrameLatencyPolicy()
        if self.name == "drop_oldest":
            return DropOldestFrameLatencyPolicy()
        if self.name == "adaptive_sampling":
            return AdaptiveSamplingFrameLatencyPolicy.from_mapping(self.params, path=f"{self._path}.params")
        raise LatencyPolicyError(
            "Unsupported latency policy "
            f"'{self.name}'. Supported values: blocking, drop_newest, drop_oldest, adaptive_sampling.",
            path=f"{self._path}.name",
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for configs and run metadata."""
        return {"name": self.name, "params": dict(self.params)}


class FrameLatencyPolicy(ABC):
    """
    Strategy used by streaming frame extractors under queue pressure.

    Implementations decide whether an incoming frame should be published,
    dropped, or used to replace an older queued frame. Instances may keep
    counters, so they are per-run runtime objects rather than reusable config
    values.
    """

    @abstractmethod
    def publish(self, frame: Frame, output_buffer: IFrameBuffer) -> bool:
        """
        Publish `frame` to `output_buffer` or drop it.

        Parameters
        ----------
        frame:
            Incoming frame from a streaming source.
        output_buffer:
            Bounded frame queue feeding downstream stages.

        Returns
        -------
        bool
            `True` when the frame was accepted by the output buffer.
        """

    def metrics(self) -> dict[str, Any]:
        """Return runtime counters for observability."""
        return {}


class BlockingFrameLatencyPolicy(FrameLatencyPolicy):
    """
    Preserve every frame by blocking upstream when the queue is full.

    Use this policy for offline reproducibility and deterministic frame
    coverage. It can increase latency in realtime pipelines when downstream
    inference or visualization is slower than the source.
    """

    def __init__(self) -> None:
        self.accepted_frames = 0

    def publish(self, frame: Frame, output_buffer: IFrameBuffer) -> bool:
        output_buffer.put(frame)
        self.accepted_frames += 1
        return True

    def metrics(self) -> dict[str, Any]:
        return {"accepted_frames": self.accepted_frames, "dropped_frames": 0}


class DropNewestFrameLatencyPolicy(FrameLatencyPolicy):
    """
    Drop the incoming frame when the downstream queue is full.

    This policy protects downstream stages from backlog growth while preserving
    already queued frames. It is useful when continuity of accepted frames is
    more important than visual freshness.
    """

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
    """
    Keep preview latency low by replacing stale queued frames.

    This policy favors the most recent source frame and is usually the best
    default for realtime camera previews where freshness is more important than
    processing every frame.
    """

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
        path: str = "run.runtime.latency_policy.params",
    ) -> None:
        if min_interval <= 0:
            raise LatencyPolicyError(
                "min_interval must be greater than 0.",
                path=f"{path}.min_interval",
            )
        if max_interval < min_interval:
            raise LatencyPolicyError(
                "max_interval must be greater than or equal to min_interval.",
                path=f"{path}.max_interval",
            )
        if not (0.0 <= low_watermark <= high_watermark <= 1.0):
            raise LatencyPolicyError(
                "watermarks must satisfy 0 <= low_watermark <= high_watermark <= 1.",
                path=path,
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
    def from_mapping(
        cls,
        params: Mapping[str, Any],
        *,
        path: str = "run.runtime.latency_policy.params",
    ) -> AdaptiveSamplingFrameLatencyPolicy:
        """
        Build an adaptive policy from config params.

        Parameters
        ----------
        params:
            Mapping with optional `min_interval`, `max_interval`,
            `high_watermark`, and `low_watermark` values.

        Raises
        ------
        LatencyPolicyError
            If numeric params cannot be coerced or violate invariants.
        """
        return cls(
            min_interval=_coerce_int(
                params.get("min_interval", 1),
                f"{path}.min_interval",
            ),
            max_interval=_coerce_int(
                params.get("max_interval", 8),
                f"{path}.max_interval",
            ),
            high_watermark=_coerce_float(
                params.get("high_watermark", 0.75),
                f"{path}.high_watermark",
            ),
            low_watermark=_coerce_float(
                params.get("low_watermark", 0.25),
                f"{path}.low_watermark",
            ),
            path=path,
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
