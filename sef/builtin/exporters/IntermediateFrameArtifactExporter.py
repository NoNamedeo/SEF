from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from library.core.artifacts.mask.MaskArtifacts import IntermediateFrameArtifact


class IntermediateFrameArtifactExporter:
    """Filesystem exporter for intermediate frame debug snapshots."""

    def __init__(self, output_directory: Path | str) -> None:
        self._output_directory = Path(output_directory)

    def export_many(self, artifacts: Iterable[IntermediateFrameArtifact]) -> tuple[Path, ...]:
        """Export every artifact and return paths in write order."""
        paths: list[Path] = []
        for artifact in artifacts:
            paths.extend(self.export(artifact))
        return tuple(paths)

    def export(self, artifact: IntermediateFrameArtifact) -> tuple[Path, ...]:
        """Export one artifact, including original, processed, masks, and overlays."""
        self._output_directory.mkdir(parents=True, exist_ok=True)
        stem = self._artifact_stem(artifact)
        written: list[Path] = []

        written.append(self._write_image(f"{stem}_snapshot.png", artifact.image, artifact.color_space))
        if artifact.original_frame is not None:
            written.append(self._write_image(f"{stem}_original.png", artifact.original_frame, artifact.color_space))
        if artifact.processed_frame is not artifact.image:
            written.append(self._write_image(f"{stem}_processed.png", artifact.processed_frame, artifact.color_space))

        for index, mask in enumerate(artifact.masks):
            label = self._safe_name(mask.label or f"mask_{index}")
            written.append(self._write_image(f"{stem}_{label}.png", mask.as_uint8_array(), None))

        for index, overlay in enumerate(artifact.overlays):
            label = self._safe_name(overlay.label or f"overlay_{index}")
            written.append(self._write_image(f"{stem}_{label}.png", overlay.image, overlay.color_space))

        return tuple(written)

    def _write_image(
        self,
        filename: str,
        image: npt.NDArray[Any],
        color_space: str | None,
    ) -> Path:
        path = self._output_directory / filename
        writable = self._to_uint8_image(image, color_space)
        if not cv2.imwrite(str(path), writable):
            raise RuntimeError(f"Failed to write intermediate frame image: {path}")
        return path

    @classmethod
    def _artifact_stem(cls, artifact: IntermediateFrameArtifact) -> str:
        frame_part = "unknown" if artifact.frame_index is None else f"{int(artifact.frame_index):06d}"
        return f"frame_{frame_part}_{cls._safe_name(artifact.stage_name)}"

    @staticmethod
    def _safe_name(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
        return normalized.strip("._") or "artifact"

    @staticmethod
    def _to_uint8_image(image: npt.NDArray[Any], color_space: str | None) -> npt.NDArray[np.uint8]:
        array = np.asarray(image)
        if array.dtype == np.bool_:
            array = array.astype(np.uint8) * 255
        elif np.issubdtype(array.dtype, np.floating):
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                array = np.zeros(array.shape, dtype=np.uint8)
            elif finite.min() >= 0.0 and finite.max() <= 1.0:
                array = np.clip(array, 0.0, 1.0) * 255.0
            else:
                minimum = float(finite.min())
                maximum = float(finite.max())
                if maximum == minimum:
                    array = np.zeros(array.shape, dtype=np.uint8)
                else:
                    array = (np.clip(array, minimum, maximum) - minimum) * (255.0 / (maximum - minimum))
            array = array.astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            array = array.copy()

        if array.ndim == 3 and array.shape[2] == 3 and str(color_space or "").upper() == "RGB":
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        return array
