from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np
import numpy.typing as npt

from library.core.artifacts.MaskOperations import MaskArray, ensure_shape_compatible, normalize_binary_mask, spatial_shape_of
from library.core.interfaces.IData import IData


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, MappingProxyType):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _readonly_image_copy(image: npt.NDArray[Any], *, name: str) -> npt.NDArray[Any]:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D or 3D; got shape {image.shape}.")
    spatial_shape_of(image, name=name)

    image_copy = np.array(image, copy=True)
    image_copy.setflags(write=False)
    return image_copy


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MaskArtifact(IData):
    """
    Immutable base for binary mask artifacts.

    The stored mask is always a read-only boolean copy. This keeps downstream
    preprocessing and debugging artifacts deterministic even when the source
    array is later reused or mutated by OpenCV code.
    """

    mask: npt.NDArray[Any] = field(repr=False)
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)

    artifact_type: ClassVar[str] = "mask"

    def __post_init__(self) -> None:
        normalized_mask = normalize_binary_mask(self.mask, name=f"{self.__class__.__name__}.mask", readonly=True)
        object.__setattr__(self, "mask", normalized_mask)
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
        object.__setattr__(self, "config", _immutable_mapping(self.config))

    @property
    def shape(self) -> tuple[int, int]:
        """Return the mask shape as `(height, width)`."""
        return int(self.mask.shape[0]), int(self.mask.shape[1])

    @property
    def height(self) -> int:
        """Return the mask height in pixels."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Return the mask width in pixels."""
        return self.shape[1]

    @property
    def active_pixel_count(self) -> int:
        """Return the number of protected, target, or active mask pixels."""
        return int(np.count_nonzero(self.mask))

    @property
    def coverage_ratio(self) -> float:
        """Return the fraction of active pixels in the mask."""
        return float(self.active_pixel_count / self.mask.size)

    @property
    def is_empty(self) -> bool:
        """Return whether the mask contains no active pixels."""
        return self.active_pixel_count == 0

    def as_bool_array(self, *, copy: bool = True) -> MaskArray:
        """
        Return the mask as a boolean NumPy array.

        By default this returns a mutable copy. Passing `copy=False` returns the
        internal read-only view for zero-copy consumers.
        """
        return self.mask.copy() if copy else self.mask

    def as_uint8_array(self, *, active_value: int = 255) -> npt.NDArray[np.uint8]:
        """Return an OpenCV-friendly uint8 mask copy."""
        if active_value < 1 or active_value > 255:
            raise ValueError("active_value must be between 1 and 255.")
        return self.mask.astype(np.uint8) * active_value

    def ensure_compatible_with(self, candidate: npt.NDArray[Any] | tuple[int, ...]) -> None:
        """Validate that a frame, shape, or other mask shares this mask's spatial dimensions."""
        ensure_shape_compatible(self.mask, candidate, reference_name=f"{self.__class__.__name__}.mask")

    def to_dict(self, *, include_mask: bool = False) -> dict[str, Any]:
        """
        Serialize lightweight artifact state for logs, tests, or debug UIs.

        The mask payload is omitted by default to avoid accidentally logging
        full-frame arrays. Set `include_mask=True` for small masks and tests.
        """
        payload: dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "shape": self.shape,
            "height": self.height,
            "width": self.width,
            "active_pixel_count": self.active_pixel_count,
            "coverage_ratio": self.coverage_ratio,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "label": self.label,
            "metadata": dict(self.metadata),
            "config": dict(self.config),
        }
        payload.update(self._extra_serialized_fields())
        if include_mask:
            payload["mask"] = self.mask.astype(np.uint8).tolist()
        return payload

    def to_json(self, *, include_mask: bool = False) -> str:
        """Serialize the artifact to deterministic JSON."""
        return json.dumps(self.to_dict(include_mask=include_mask), default=_json_default, sort_keys=True)

    def debug_string(self) -> str:
        """Return a compact, array-safe debug representation."""
        label = f", label={self.label!r}" if self.label is not None else ""
        return (
            f"{self.__class__.__name__}(type={self.artifact_type!r}, shape={self.shape}, "
            f"active_pixels={self.active_pixel_count}, coverage={self.coverage_ratio:.4f}, "
            f"frame_index={self.frame_index!r}{label})"
        )

    def _extra_serialized_fields(self) -> dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return self.debug_string()

    def __repr__(self) -> str:
        return self.debug_string()


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class FrameMaskArtifact(MaskArtifact):
    """Whole-frame binary mask produced or consumed by preprocessing stages."""

    artifact_type: ClassVar[str] = "frame_mask"


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MotionMaskArtifact(MaskArtifact):
    """Binary mask representing detected motion pixels for a frame."""

    artifact_type: ClassVar[str] = "motion_mask"


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class TargetMaskArtifact(MaskArtifact):
    """Binary mask that identifies the target subject or region for future workflows."""

    target_id: str | None = None

    artifact_type: ClassVar[str] = "target_mask"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.target_id is not None and not self.target_id.strip():
            raise ValueError("TargetMaskArtifact.target_id cannot be empty when provided.")

    def _extra_serialized_fields(self) -> dict[str, Any]:
        return {"target_id": self.target_id}


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ProtectedRegionArtifact(MaskArtifact):
    """Binary mask describing pixels that downstream stages should leave untouched."""

    region_id: str | None = None
    reason: str | None = None

    artifact_type: ClassVar[str] = "protected_region"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.region_id is not None and not self.region_id.strip():
            raise ValueError("ProtectedRegionArtifact.region_id cannot be empty when provided.")
        if self.reason is not None and not self.reason.strip():
            raise ValueError("ProtectedRegionArtifact.reason cannot be empty when provided.")

    def _extra_serialized_fields(self) -> dict[str, Any]:
        return {"region_id": self.region_id, "reason": self.reason}


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class IntermediateFrameOverlay:
    """
    Immutable rendered overlay associated with an intermediate frame snapshot.

    Overlays are intentionally image-based rather than UI objects. This keeps
    frame cleaners free to emit masks, contours, heatmaps, or other debug
    images without coupling them to a specific visualization backend.
    """

    image: npt.NDArray[Any] = field(repr=False)
    label: str | None = None
    color_space: str | None = None
    alpha: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label is not None and not self.label.strip():
            raise ValueError("IntermediateFrameOverlay.label cannot be empty when provided.")
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("IntermediateFrameOverlay.alpha must be between 0.0 and 1.0.")
        object.__setattr__(
            self,
            "image",
            _readonly_image_copy(self.image, name="IntermediateFrameOverlay.image"),
        )
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the full overlay image shape."""
        return tuple(int(dimension) for dimension in self.image.shape)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return the overlay shape as `(height, width)`."""
        return spatial_shape_of(self.image, name="IntermediateFrameOverlay.image")

    def as_array(self, *, copy: bool = True) -> npt.NDArray[Any]:
        """Return the overlay image, copying by default for caller safety."""
        return self.image.copy() if copy else self.image


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class IntermediateFrameArtifact(IData):
    """
    Immutable snapshot of an intermediate frame for pipeline debugging.

    The image is copied and marked read-only so snapshots represent the exact
    state emitted by a stage, independent of later OpenCV buffer reuse.
    """

    image: npt.NDArray[Any] = field(repr=False)
    stage_name: str
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    color_space: str | None = None
    original_image: npt.NDArray[Any] | None = field(default=None, repr=False)
    cleaned_image: npt.NDArray[Any] | None = field(default=None, repr=False)
    masks: Sequence[MaskArtifact] = field(default_factory=tuple)
    overlays: Sequence[IntermediateFrameOverlay] = field(default_factory=tuple)
    stage_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)

    artifact_type: ClassVar[str] = "intermediate_frame"

    def __post_init__(self) -> None:
        if not self.stage_name.strip():
            raise ValueError("IntermediateFrameArtifact.stage_name must be a non-empty string.")

        image_copy = _readonly_image_copy(self.image, name="IntermediateFrameArtifact.image")
        object.__setattr__(self, "image", image_copy)
        if self.original_image is not None:
            object.__setattr__(
                self,
                "original_image",
                _readonly_image_copy(
                    self.original_image,
                    name="IntermediateFrameArtifact.original_image",
                ),
            )
        if self.cleaned_image is not None:
            object.__setattr__(
                self,
                "cleaned_image",
                _readonly_image_copy(
                    self.cleaned_image,
                    name="IntermediateFrameArtifact.cleaned_image",
                ),
            )

        masks = tuple(self.masks)
        if any(not isinstance(mask, MaskArtifact) for mask in masks):
            raise TypeError("IntermediateFrameArtifact.masks must contain MaskArtifact instances.")
        for mask in masks:
            self.ensure_mask_compatible(mask)
        object.__setattr__(self, "masks", masks)

        overlays = tuple(self.overlays)
        if any(not isinstance(overlay, IntermediateFrameOverlay) for overlay in overlays):
            raise TypeError("IntermediateFrameArtifact.overlays must contain IntermediateFrameOverlay instances.")
        for overlay in overlays:
            ensure_shape_compatible(
                image_copy,
                overlay.image,
                reference_name="IntermediateFrameArtifact.image",
                candidate_name="overlay",
            )
        object.__setattr__(self, "overlays", overlays)
        object.__setattr__(self, "stage_metadata", _immutable_mapping(self.stage_metadata))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
        object.__setattr__(self, "config", _immutable_mapping(self.config))

    @property
    def frame(self) -> npt.NDArray[Any]:
        """Alias matching the existing `Frame.frame` convention."""
        return self.image

    @property
    def original_frame(self) -> npt.NDArray[Any] | None:
        """Return the optional pre-cleaning frame snapshot."""
        return self.original_image

    @property
    def cleaned_frame(self) -> npt.NDArray[Any]:
        """Return the cleaned frame snapshot, falling back to the primary image."""
        return self.cleaned_image if self.cleaned_image is not None else self.image

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the full image shape."""
        return tuple(int(dimension) for dimension in self.image.shape)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Return the image shape as `(height, width)`."""
        return spatial_shape_of(self.image, name="IntermediateFrameArtifact.image")

    @property
    def height(self) -> int:
        """Return the frame height in pixels."""
        return self.spatial_shape[0]

    @property
    def width(self) -> int:
        """Return the frame width in pixels."""
        return self.spatial_shape[1]

    @property
    def channels(self) -> int:
        """Return the number of image channels, treating grayscale frames as one-channel."""
        return 1 if self.image.ndim == 2 else int(self.image.shape[2])

    @property
    def dtype(self) -> np.dtype[Any]:
        """Return the NumPy dtype of the stored snapshot."""
        return self.image.dtype

    def as_array(self, *, copy: bool = True) -> npt.NDArray[Any]:
        """
        Return the image snapshot.

        By default this returns a mutable copy. Passing `copy=False` returns the
        internal read-only array for zero-copy debug consumers.
        """
        return self.image.copy() if copy else self.image

    def ensure_mask_compatible(self, mask: MaskArtifact | npt.NDArray[Any] | tuple[int, ...]) -> None:
        """Validate that a mask-like object can be applied to this frame snapshot."""
        candidate = mask.mask if isinstance(mask, MaskArtifact) else mask
        ensure_shape_compatible(self.image, candidate, reference_name="IntermediateFrameArtifact.image", candidate_name="mask")

    def to_dict(self, *, include_image: bool = False) -> dict[str, Any]:
        """
        Serialize lightweight frame snapshot state.

        Pixel data is omitted by default because intermediate frames can be
        large. Set `include_image=True` only for small fixtures or local debug.
        """
        payload: dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "stage_name": self.stage_name,
            "shape": self.shape,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "dtype": str(self.dtype),
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "color_space": self.color_space,
            "has_original_frame": self.original_image is not None,
            "has_cleaned_frame": self.cleaned_image is not None,
            "mask_count": len(self.masks),
            "overlay_count": len(self.overlays),
            "stage_metadata": dict(self.stage_metadata),
            "metadata": dict(self.metadata),
            "config": dict(self.config),
        }
        if include_image:
            payload["image"] = self.image.tolist()
        return payload

    def to_json(self, *, include_image: bool = False) -> str:
        """Serialize the snapshot to deterministic JSON."""
        return json.dumps(self.to_dict(include_image=include_image), default=_json_default, sort_keys=True)

    def debug_string(self) -> str:
        """Return a compact, pixel-safe debug representation."""
        return (
            f"{self.__class__.__name__}(stage_name={self.stage_name!r}, shape={self.shape}, "
            f"dtype={self.dtype}, frame_index={self.frame_index!r}, color_space={self.color_space!r})"
        )

    def __str__(self) -> str:
        return self.debug_string()

    def __repr__(self) -> str:
        return self.debug_string()
