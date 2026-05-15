from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DEFAULT_MAX_SEQUENCE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_SAMPLE_STACK_BYTES = 512 * 1024 * 1024
HARD_MAX_PROCESSED_FRAMES = 10_000


@dataclass(frozen=True, slots=True)
class ProtectedRegion:
    """Rectangular frame region that must not be altered by object removal."""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | Sequence[int]) -> "ProtectedRegion":
        """Parse a protected region from a config mapping or ``(x, y, width, height)`` sequence."""
        if isinstance(value, Mapping):
            region = cls(
                x=int(value["x"]),
                y=int(value["y"]),
                width=int(value["width"]),
                height=int(value["height"]),
            )
        else:
            if len(value) != 4:
                raise ValueError("Protected region sequences must contain exactly four values: x, y, width, height.")
            x, y, width, height = value
            region = cls(x=int(x), y=int(y), width=int(width), height=int(height))
        region.validate()
        return region

    def validate(self) -> None:
        """Validate basic rectangle invariants before clipping to frame bounds."""
        if self.x < 0 or self.y < 0:
            raise ValueError("Protected region x and y must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Protected region width and height must be greater than 0.")


@dataclass(frozen=True, slots=True)
class DynamicObjectRemovalConfig:
    """
    Configuration for temporal background based dynamic object removal.

    The defaults are intentionally conservative for local machines: full-frame
    temporal algorithms can allocate large sample stacks even before producing
    output frames.
    """

    sampling_stride: int = 5
    max_sampled_frames: int = 60
    difference_threshold: float = 35.0
    difference_mode: str = "luma"
    morph_kernel_size: int = 5
    opening_iterations: int = 1
    closing_iterations: int = 2
    dilation_iterations: int = 1
    min_component_area: int = 80
    max_processed_frames: int = 300
    max_sequence_bytes: int = DEFAULT_MAX_SEQUENCE_BYTES
    max_sample_stack_bytes: int = DEFAULT_MAX_SAMPLE_STACK_BYTES
    emit_intermediate_artifacts: bool = False

    def __post_init__(self) -> None:
        if self.sampling_stride <= 0:
            raise ValueError("sampling_stride must be greater than 0.")
        if self.max_sampled_frames <= 0:
            raise ValueError("max_sampled_frames must be greater than 0.")
        if self.difference_threshold < 0:
            raise ValueError("difference_threshold cannot be negative.")
        if self.difference_mode not in {"luma", "mean", "max"}:
            raise ValueError("difference_mode must be one of: luma, mean, max.")
        if self.morph_kernel_size <= 0:
            raise ValueError("morph_kernel_size must be greater than 0.")
        if self.morph_kernel_size % 2 == 0:
            raise ValueError("morph_kernel_size must be odd to keep morphology centered.")
        if self.opening_iterations < 0:
            raise ValueError("opening_iterations cannot be negative.")
        if self.closing_iterations < 0:
            raise ValueError("closing_iterations cannot be negative.")
        if self.dilation_iterations < 0:
            raise ValueError("dilation_iterations cannot be negative.")
        if self.min_component_area < 0:
            raise ValueError("min_component_area cannot be negative.")
        if self.max_processed_frames <= 0:
            raise ValueError("max_processed_frames must be greater than 0.")
        if self.max_processed_frames > HARD_MAX_PROCESSED_FRAMES:
            raise ValueError(f"max_processed_frames cannot exceed hard limit {HARD_MAX_PROCESSED_FRAMES}.")
        if self.max_sequence_bytes <= 0:
            raise ValueError("max_sequence_bytes must be greater than 0.")
        if self.max_sample_stack_bytes <= 0:
            raise ValueError("max_sample_stack_bytes must be greater than 0.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly config snapshot for frame metadata and artifacts."""
        return {
            "sampling_stride": self.sampling_stride,
            "max_sampled_frames": self.max_sampled_frames,
            "difference_threshold": self.difference_threshold,
            "difference_mode": self.difference_mode,
            "morph_kernel_size": self.morph_kernel_size,
            "opening_iterations": self.opening_iterations,
            "closing_iterations": self.closing_iterations,
            "dilation_iterations": self.dilation_iterations,
            "min_component_area": self.min_component_area,
            "max_processed_frames": self.max_processed_frames,
            "max_sequence_bytes": self.max_sequence_bytes,
            "max_sample_stack_bytes": self.max_sample_stack_bytes,
            "emit_intermediate_artifacts": self.emit_intermediate_artifacts,
        }
