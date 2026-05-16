from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.MaskArtifacts import (
    FrameMaskArtifact,
    IntermediateFrameArtifact,
    IntermediateFrameOverlay,
    MotionMaskArtifact,
    ProtectedRegionArtifact,
)
from library.core.artifacts.MaskOperations import (
    ensure_shape_compatible,
    normalize_binary_mask,
    subtract_masks,
)
from library.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.pipeline.FrameProcessingStage import FrameProcessorExecutionContext
from library.frame_processors.dynamic_object_removal.background_estimator import (
    BackgroundEstimator,
    TemporalMedianBackgroundEstimator,
)
from library.frame_processors.dynamic_object_removal.config import (
    DynamicObjectRemovalConfig,
    ProtectedRegion,
)
from library.frame_processors.dynamic_object_removal.foreground_mask_extractor import (
    BackgroundDifferenceForegroundMaskExtractor,
    ForegroundMaskExtractor,
)
from library.frame_processors.dynamic_object_removal.mask_refiner import MaskRefiner, MaskRefinementResult, MorphologicalMaskRefiner
from library.frame_processors.dynamic_object_removal.region_reconstructor import (
    BackgroundReplacementRegionReconstructor,
    RegionReconstructor,
)

_METADATA_KEY = "dynamic_object_removal"
_PROGRESS_INTERVAL = 25
log = logging.getLogger(__name__)


class DynamicObjectRemovalFrameProcessor(IFrameBufferProcessor):
    """
    Remove transient dynamic objects by replacing foreground pixels with a temporal median background.

    This processor is intentionally offline and sequence-aware: it consumes the
    frame sequence, estimates a static background, and only then reconstructs
    each frame. It is therefore registered as a frame-buffer processor, not as a
    streaming single-frame processor.
    """

    capabilities = StageCapabilities.batch(
        stateful=True,
        realtime_safe=False,
    )

    def __init__(
        self,
        sampling_stride: int = 5,
        max_sampled_frames: int = 60,
        difference_threshold: float = 35.0,
        difference_mode: str = "luma",
        morph_kernel_size: int = 5,
        opening_iterations: int = 1,
        closing_iterations: int = 2,
        dilation_iterations: int = 1,
        min_component_area: int = 80,
        max_processed_frames: int = 300,
        max_sequence_bytes: int = 512 * 1024 * 1024,
        max_sample_stack_bytes: int = 512 * 1024 * 1024,
        protected_mask: npt.NDArray | ProtectedRegionArtifact | None = None,
        protected_regions: Sequence[Mapping[str, Any] | Sequence[int]] | None = None,
        emit_intermediate_artifacts: bool = False,
        background_estimator: BackgroundEstimator | None = None,
        foreground_mask_extractor: ForegroundMaskExtractor | None = None,
        mask_refiner: MaskRefiner | None = None,
        region_reconstructor: RegionReconstructor | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        removal_config = DynamicObjectRemovalConfig(
            sampling_stride=sampling_stride,
            max_sampled_frames=max_sampled_frames,
            difference_threshold=difference_threshold,
            difference_mode=difference_mode,
            morph_kernel_size=morph_kernel_size,
            opening_iterations=opening_iterations,
            closing_iterations=closing_iterations,
            dilation_iterations=dilation_iterations,
            min_component_area=min_component_area,
            max_processed_frames=max_processed_frames,
            max_sequence_bytes=max_sequence_bytes,
            max_sample_stack_bytes=max_sample_stack_bytes,
            emit_intermediate_artifacts=emit_intermediate_artifacts,
        )
        super().__init__({**removal_config.as_dict(), **dict(config or {})})
        self.removal_config = removal_config
        self.protected_mask = protected_mask
        self.protected_regions = tuple(ProtectedRegion.from_value(region) for region in (protected_regions or ()))
        self.background_estimator = background_estimator or TemporalMedianBackgroundEstimator(removal_config)
        self.foreground_mask_extractor = foreground_mask_extractor or BackgroundDifferenceForegroundMaskExtractor(removal_config)
        self.mask_refiner = mask_refiner or MorphologicalMaskRefiner(removal_config)
        self.region_reconstructor = region_reconstructor or BackgroundReplacementRegionReconstructor()

    def process(self, buffer: FrameBuffer) -> FrameBuffer:
        """Process a frame sequence without emitting intermediate artifacts."""
        return self._process(buffer, context=None)

    def process_with_context(
        self,
        buffer: FrameBuffer,
        context: FrameProcessorExecutionContext,
    ) -> FrameBuffer:
        """Process a frame sequence and emit bounded debug artifacts when capture is enabled."""
        return self._process(buffer, context=context)

    def _process(
        self,
        buffer: FrameBuffer,
        *,
        context: FrameProcessorExecutionContext | None,
    ) -> FrameBuffer:
        log.info("Dynamic object removal: reading frame sequence.")
        frames = self._read_and_validate_frames(buffer)
        log.info(
            "Dynamic object removal: loaded %s frames, shape=%s, dtype=%s.",
            len(frames),
            frames[0].image.shape,
            frames[0].image.dtype,
        )
        background = self.background_estimator.estimate(frames)
        protected_mask = self._build_protected_mask(background.shape)
        if protected_mask is not None:
            log.info("Dynamic object removal: protected pixels=%s.", int(np.count_nonzero(protected_mask)))

        output = FrameBuffer(buffer_size=max(len(frames) + 1, buffer.capacity))
        for source_sequence_index, frame in enumerate(frames):
            processed_frame = self._process_frame(
                frame=frame,
                background=background,
                protected_mask=protected_mask,
                source_sequence_index=source_sequence_index,
                context=context,
            )
            output.put(processed_frame)
            self._log_progress(source_sequence_index, len(frames), processed_frame)
        output.close()
        log.info("Dynamic object removal: completed %s frames.", len(frames))
        return output

    def _read_and_validate_frames(self, buffer: FrameBuffer) -> list[Frame]:
        frames: list[Frame] = []
        reference_shape: tuple[int, ...] | None = None
        reference_dtype: np.dtype | None = None
        total_bytes = 0

        for frame in buffer:
            self._validate_frame(frame)
            image = frame.image
            if reference_shape is None:
                reference_shape = tuple(int(value) for value in image.shape)
                reference_dtype = image.dtype
            elif tuple(image.shape) != reference_shape:
                raise ValueError(f"Frame {frame.index} shape {image.shape} does not match first frame shape {reference_shape}.")
            elif image.dtype != reference_dtype:
                raise ValueError(f"Frame {frame.index} dtype {image.dtype} does not match first frame dtype {reference_dtype}.")

            frames.append(frame)
            if len(frames) > self.removal_config.max_processed_frames:
                raise ValueError(
                    "Dynamic object removal would process more than "
                    f"{self.removal_config.max_processed_frames} frames. Lower extractor max_frames/stride or "
                    "increase max_processed_frames explicitly."
                )

            total_bytes += int(image.nbytes)
            if total_bytes > self.removal_config.max_sequence_bytes:
                raise ValueError(
                    "Dynamic object removal frame sequence would exceed "
                    f"{self.removal_config.max_sequence_bytes} bytes. Resize the video, reduce max_frames, "
                    "or increase max_sequence_bytes explicitly."
                )

        if not frames:
            raise ValueError("DynamicObjectRemovalFrameProcessor requires at least one frame.")
        return frames

    @staticmethod
    def _log_progress(source_sequence_index: int, frame_count: int, frame: Frame) -> None:
        processed_count = source_sequence_index + 1
        if processed_count != frame_count and processed_count % _PROGRESS_INTERVAL != 0:
            return

        metrics = dict(frame.metadata.get(_METADATA_KEY, {}))
        log.info(
            "Dynamic object removal: processed %s/%s frames, removed_pixels=%s, removed_ratio=%.4f.",
            processed_count,
            frame_count,
            metrics.get("removed_pixel_count", 0),
            float(metrics.get("removed_pixel_ratio", 0.0)),
        )

    @staticmethod
    def _validate_frame(frame: Frame) -> None:
        if not isinstance(frame.image, np.ndarray):
            raise TypeError("Frame.image must be a numpy.ndarray.")
        if frame.image.ndim not in (2, 3):
            raise ValueError(f"Frame.image must be 2D or 3D; got shape {frame.image.shape}.")
        if frame.image.size == 0:
            raise ValueError("Frame.image cannot be empty.")
        if not np.issubdtype(frame.image.dtype, np.number):
            raise TypeError(f"Frame.image dtype must be numeric; got {frame.image.dtype}.")
        if np.issubdtype(frame.image.dtype, np.floating) and not np.isfinite(frame.image).all():
            raise ValueError("Frame.image contains NaN or infinite values.")

    def _build_protected_mask(self, frame_shape: tuple[int, ...]) -> npt.NDArray[np.bool_] | None:
        protected_masks: list[npt.NDArray[np.bool_]] = []
        if self.protected_mask is not None:
            if isinstance(self.protected_mask, ProtectedRegionArtifact):
                mask = self.protected_mask.as_bool_array(copy=True)
            else:
                mask = normalize_binary_mask(self.protected_mask, name="protected_mask")
            ensure_shape_compatible(frame_shape, mask, reference_name="frame", candidate_name="protected_mask")
            protected_masks.append(mask)

        if self.protected_regions:
            protected_masks.append(self._protected_regions_to_mask(frame_shape))

        if not protected_masks:
            return None

        merged = np.zeros(protected_masks[0].shape, dtype=np.bool_)
        for mask in protected_masks:
            ensure_shape_compatible(merged, mask, reference_name="merged_protected_mask", candidate_name="protected_mask")
            np.logical_or(merged, mask, out=merged)
        return merged

    def _protected_regions_to_mask(self, frame_shape: tuple[int, ...]) -> npt.NDArray[np.bool_]:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        mask = np.zeros((height, width), dtype=np.bool_)
        for region in self.protected_regions:
            x1 = min(region.x, width)
            y1 = min(region.y, height)
            x2 = min(region.x + region.width, width)
            y2 = min(region.y + region.height, height)
            if x1 < x2 and y1 < y2:
                mask[y1:y2, x1:x2] = True
        return mask

    def _process_frame(
        self,
        *,
        frame: Frame,
        background: npt.NDArray,
        protected_mask: npt.NDArray[np.bool_] | None,
        source_sequence_index: int,
        context: FrameProcessorExecutionContext | None,
    ) -> Frame:
        raw_mask = self.foreground_mask_extractor.extract(frame.image, background)
        refinement = self.mask_refiner.refine(raw_mask)
        effective_mask = (
            subtract_masks(refinement.mask, protected_mask)
            if protected_mask is not None
            else refinement.mask
        )
        cleaned = self.region_reconstructor.reconstruct(frame.image, background, effective_mask)
        metadata = self._frame_metadata(frame, effective_mask, refinement)
        processed = Frame(
            image=cleaned,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata={**dict(frame.metadata), _METADATA_KEY: metadata},
        )
        self._capture_intermediate_artifact(
            original_frame=frame,
            processed_frame=processed,
            background=background,
            raw_mask=raw_mask,
            refined_mask=refinement.mask,
            effective_mask=effective_mask,
            protected_mask=protected_mask,
            source_sequence_index=source_sequence_index,
            context=context,
            metrics=metadata,
        )
        return processed

    def _frame_metadata(
        self,
        frame: Frame,
        effective_mask: npt.NDArray[np.bool_],
        refinement: MaskRefinementResult,
    ) -> dict[str, Any]:
        removed_pixels = int(np.count_nonzero(effective_mask))
        total_pixels = int(effective_mask.size)
        return {
            "frame_index": frame.index,
            "removed_pixel_count": removed_pixels,
            "removed_pixel_ratio": float(removed_pixels / total_pixels) if total_pixels else 0.0,
            "component_count": refinement.component_count,
            "removed_component_count": refinement.removed_component_count,
            "average_component_area": refinement.average_component_area,
            "sampled_indices": list(getattr(self.background_estimator, "last_sampled_indices", ())),
        }

    def _capture_intermediate_artifact(
        self,
        *,
        original_frame: Frame,
        processed_frame: Frame,
        background: npt.NDArray,
        raw_mask: npt.NDArray[np.bool_],
        refined_mask: npt.NDArray[np.bool_],
        effective_mask: npt.NDArray[np.bool_],
        protected_mask: npt.NDArray[np.bool_] | None,
        source_sequence_index: int,
        context: FrameProcessorExecutionContext | None,
        metrics: Mapping[str, Any],
    ) -> None:
        if context is None or context.intermediate_store is None:
            return
        if not context.intermediate_store.should_capture(source_sequence_index):
            return
        if not (self.removal_config.emit_intermediate_artifacts or context.intermediate_store.config.enabled):
            return

        masks = [
            MotionMaskArtifact(mask=raw_mask, frame_index=processed_frame.index, label="raw_dynamic_mask"),
            MotionMaskArtifact(mask=refined_mask, frame_index=processed_frame.index, label="refined_dynamic_mask"),
            FrameMaskArtifact(mask=effective_mask, frame_index=processed_frame.index, label="effective_removal_mask"),
        ]
        if protected_mask is not None:
            masks.append(
                ProtectedRegionArtifact(
                    mask=protected_mask,
                    frame_index=processed_frame.index,
                    label="protected_mask",
                    reason="dynamic_object_removal_protection",
                )
            )

        context.intermediate_store.add(
            IntermediateFrameArtifact(
                image=processed_frame.image,
                stage_name=context.stage_name,
                frame_index=processed_frame.index,
                timestamp_seconds=processed_frame.timestamp_seconds,
                original_image=original_frame.image if context.intermediate_store.config.include_original else None,
                processed_image=processed_frame.image,
                masks=tuple(masks),
                overlays=(
                    IntermediateFrameOverlay(
                        image=background,
                        label="estimated_background",
                        alpha=1.0,
                    ),
                ),
                stage_metadata={
                    "source_sequence_index": source_sequence_index,
                    "processor_name": context.processor_name,
                    "metrics": dict(metrics),
                },
                metadata=dict(metrics),
                config=self.removal_config.as_dict(),
            ),
            source_sequence_index=source_sequence_index,
        )
