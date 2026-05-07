from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from library.core.artifacts.MaskArtifacts import IntermediateFrameArtifact, MaskArtifact


ColorBGR = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class FrameComparisonPanel:
    """A normalized image panel used by comparison and grid composers."""

    image: npt.NDArray[Any]
    label: str | None = None
    color_space: str | None = None


def compose_intermediate_frame_comparison(
    artifact: IntermediateFrameArtifact,
    *,
    show_labels: bool = True,
    include_masks: bool = True,
    include_overlays: bool = True,
    max_panel_width: int | None = 480,
    gap: int = 8,
    background_color: ColorBGR = (20, 24, 28),
) -> npt.NDArray[np.uint8]:
    """
    Compose one intermediate artifact as a side-by-side debug comparison.

    The output is a BGR uint8 image ready for OpenCV encoding.
    """
    panels: list[FrameComparisonPanel] = []
    if artifact.original_frame is not None:
        panels.append(
            FrameComparisonPanel(
                image=artifact.original_frame,
                label="Original",
                color_space=artifact.color_space,
            )
        )

    panels.append(
        FrameComparisonPanel(
            image=artifact.processed_frame,
            label=artifact.stage_name,
            color_space=artifact.color_space,
        )
    )

    if include_masks:
        for mask in artifact.masks:
            panels.append(
                FrameComparisonPanel(
                    image=apply_mask_overlay(artifact.processed_frame, mask),
                    label=mask.label or "Mask overlay",
                )
            )

    if include_overlays:
        for overlay in artifact.overlays:
            panels.append(
                FrameComparisonPanel(
                    image=blend_overlay(
                        artifact.processed_frame,
                        overlay.image,
                        overlay_alpha=overlay.alpha,
                        base_color_space=artifact.color_space,
                        overlay_color_space=overlay.color_space,
                    ),
                    label=overlay.label or "Overlay",
                )
            )

    return compose_side_by_side(
        panels,
        show_labels=show_labels,
        max_panel_width=max_panel_width,
        gap=gap,
        background_color=background_color,
    )


def compose_side_by_side(
    panels: Sequence[FrameComparisonPanel] | Sequence[npt.NDArray[Any]],
    *,
    labels: Sequence[str | None] | None = None,
    color_space: str | None = None,
    show_labels: bool = True,
    max_panel_width: int | None = None,
    gap: int = 8,
    background_color: ColorBGR = (20, 24, 28),
) -> npt.NDArray[np.uint8]:
    """Compose images horizontally with optional labels."""
    normalized_panels = _normalize_panels(panels, labels=labels, color_space=color_space)
    if not normalized_panels:
        raise ValueError("compose_side_by_side requires at least one image panel.")
    images = [
        _with_label(
            _resize_to_max_width(to_display_bgr(panel.image, panel.color_space), max_panel_width),
            panel.label if show_labels else None,
            background_color,
        )
        for panel in normalized_panels
    ]
    return _compose_rows([images], gap=gap, background_color=background_color)


def compose_image_grid(
    images: Sequence[npt.NDArray[Any]],
    *,
    labels: Sequence[str | None] | None = None,
    columns: int = 2,
    max_cell_width: int | None = 640,
    gap: int = 10,
    background_color: ColorBGR = (20, 24, 28),
) -> npt.NDArray[np.uint8]:
    """Compose BGR/RGB/grayscale images into a padded grid."""
    if columns <= 0:
        raise ValueError("columns must be greater than 0.")
    if not images:
        raise ValueError("compose_image_grid requires at least one image.")

    labels = labels or (None,) * len(images)
    if len(labels) != len(images):
        raise ValueError("labels must have the same length as images.")

    cells = [
        _with_label(
            _resize_to_max_width(to_display_bgr(image), max_cell_width),
            label,
            background_color,
        )
        for image, label in zip(images, labels)
    ]
    rows = [cells[index : index + columns] for index in range(0, len(cells), columns)]
    return _compose_rows(rows, gap=gap, background_color=background_color)


def apply_mask_overlay(
    image: npt.NDArray[Any],
    mask: MaskArtifact | npt.NDArray[Any],
    *,
    color: ColorBGR = (0, 220, 255),
    alpha: float = 0.35,
    color_space: str | None = None,
) -> npt.NDArray[np.uint8]:
    """Return a copy of *image* with a colored binary mask overlay."""
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")
    base = to_display_bgr(image, color_space)
    mask_array = mask.as_bool_array(copy=False) if isinstance(mask, MaskArtifact) else np.asarray(mask).astype(bool)
    if mask_array.shape[:2] != base.shape[:2]:
        raise ValueError("Mask spatial shape must match image spatial shape.")

    overlay = base.copy()
    color_array = np.array(color, dtype=np.float32)
    overlay[mask_array] = (
        (1.0 - alpha) * overlay[mask_array].astype(np.float32) + alpha * color_array
    ).astype(np.uint8)
    return overlay


def blend_overlay(
    base_image: npt.NDArray[Any],
    overlay_image: npt.NDArray[Any],
    *,
    overlay_alpha: float,
    base_color_space: str | None = None,
    overlay_color_space: str | None = None,
) -> npt.NDArray[np.uint8]:
    """Blend a rendered overlay image onto a base frame."""
    if overlay_alpha < 0.0 or overlay_alpha > 1.0:
        raise ValueError("overlay_alpha must be between 0.0 and 1.0.")
    base = to_display_bgr(base_image, base_color_space)
    overlay = to_display_bgr(overlay_image, overlay_color_space)
    if overlay.shape[:2] != base.shape[:2]:
        raise ValueError("Overlay spatial shape must match base image spatial shape.")
    return cv2.addWeighted(base, 1.0 - overlay_alpha, overlay, overlay_alpha, 0.0)


def encode_png(image: npt.NDArray[Any], *, color_space: str | None = None) -> bytes:
    """Encode an image as PNG bytes."""
    bgr = to_display_bgr(image, color_space)
    success, encoded = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("Failed to encode PNG artifact.")
    return bytes(encoded)


def to_display_bgr(
    image: npt.NDArray[Any],
    color_space: str | None = None,
) -> npt.NDArray[np.uint8]:
    """Normalize grayscale/RGB/BGR arrays to BGR uint8 for composition."""
    array = _to_uint8(np.asarray(image))
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {array.shape}.")

    channels = array.shape[2]
    if channels == 1:
        return cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
    if channels == 3:
        if str(color_space or "").upper() == "RGB":
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        return array.copy()
    if channels == 4:
        if str(color_space or "").upper() == "RGBA":
            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported channel count for display image: {channels}.")


def _to_uint8(array: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return np.zeros(array.shape, dtype=np.uint8)
        minimum = float(finite.min())
        maximum = float(finite.max())
        if minimum >= 0.0 and maximum <= 1.0:
            return (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        if maximum == minimum:
            return np.zeros(array.shape, dtype=np.uint8)
        return ((np.clip(array, minimum, maximum) - minimum) * (255.0 / (maximum - minimum))).astype(np.uint8)
    if array.dtype == np.uint8:
        return array.copy()
    return np.clip(array, 0, 255).astype(np.uint8)


def _normalize_panels(
    panels: Sequence[FrameComparisonPanel] | Sequence[npt.NDArray[Any]],
    *,
    labels: Sequence[str | None] | None,
    color_space: str | None,
) -> tuple[FrameComparisonPanel, ...]:
    if not panels:
        return ()
    if isinstance(panels[0], FrameComparisonPanel):
        return tuple(panels)  # type: ignore[arg-type]
    labels = labels or (None,) * len(panels)
    if len(labels) != len(panels):
        raise ValueError("labels must have the same length as panels.")
    return tuple(
        FrameComparisonPanel(image=image, label=label, color_space=color_space)
        for image, label in zip(panels, labels)
    )


def _resize_to_max_width(
    image: npt.NDArray[np.uint8],
    max_width: int | None,
) -> npt.NDArray[np.uint8]:
    if max_width is None or image.shape[1] <= max_width:
        return image
    if max_width <= 0:
        raise ValueError("max_width must be greater than 0 when provided.")
    scale = max_width / float(image.shape[1])
    height = max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)


def _with_label(
    image: npt.NDArray[np.uint8],
    label: str | None,
    background_color: ColorBGR,
) -> npt.NDArray[np.uint8]:
    if not label:
        return image
    label_height = 28
    canvas = np.full(
        (image.shape[0] + label_height, image.shape[1], 3),
        background_color,
        dtype=np.uint8,
    )
    canvas[label_height:, :, :] = image
    cv2.putText(
        canvas,
        label,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (236, 241, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _compose_rows(
    rows: Sequence[Sequence[npt.NDArray[np.uint8]]],
    *,
    gap: int,
    background_color: ColorBGR,
) -> npt.NDArray[np.uint8]:
    if gap < 0:
        raise ValueError("gap cannot be negative.")
    row_images = [_compose_row(row, gap=gap, background_color=background_color) for row in rows if row]
    if not row_images:
        raise ValueError("At least one image row is required.")

    width = max(image.shape[1] for image in row_images)
    height = sum(image.shape[0] for image in row_images) + gap * (len(row_images) - 1)
    canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
    y = 0
    for row in row_images:
        canvas[y : y + row.shape[0], : row.shape[1], :] = row
        y += row.shape[0] + gap
    return canvas


def _compose_row(
    images: Sequence[npt.NDArray[np.uint8]],
    *,
    gap: int,
    background_color: ColorBGR,
) -> npt.NDArray[np.uint8]:
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
    x = 0
    for image in images:
        canvas[: image.shape[0], x : x + image.shape[1], :] = image
        x += image.shape[1] + gap
    return canvas
