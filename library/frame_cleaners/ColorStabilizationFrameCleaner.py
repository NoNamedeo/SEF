from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import cv2
import numpy as np

from library.core.artifacts.Frame import Frame
from library.core.interfaces.IFrameCleaner import IFrameCleaner

_MAX_CHANNEL_VALUE = 255.0
_EPSILON = 1e-6
_METADATA_KEY = "color_stabilization"
_DEFAULT_TECHNIQUES = ("luminance_normalization", "temporal_smoothing")


class ColorSpace(StrEnum):
    """Color spaces supported by the stabilization cleaner."""

    RGB = "RGB"
    HSV = "HSV"
    LAB = "LAB"
    YCRCB = "YCrCb"

    @classmethod
    def from_value(cls, value: str | "ColorSpace") -> "ColorSpace":
        normalized = str(value).strip().replace("_", "").replace("-", "").upper()
        aliases = {
            "RGB": cls.RGB,
            "HSV": cls.HSV,
            "LAB": cls.LAB,
            "YCRCB": cls.YCRCB,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"Unsupported color_space '{value}'. Allowed values: {allowed}.") from exc


class StabilizationTechnique(StrEnum):
    """Named processing steps that can be enabled independently."""

    HISTOGRAM_NORMALIZATION = "histogram_normalization"
    CLAHE = "clahe"
    LUMINANCE_NORMALIZATION = "luminance_normalization"
    TEMPORAL_SMOOTHING = "temporal_smoothing"
    GAMMA_CORRECTION = "gamma_correction"

    @classmethod
    def from_value(cls, value: str | "StabilizationTechnique") -> "StabilizationTechnique":
        normalized = str(value).strip().replace("-", "_").lower()
        aliases = {
            "histogram": cls.HISTOGRAM_NORMALIZATION,
            "histogram_normalization": cls.HISTOGRAM_NORMALIZATION,
            "normalization": cls.HISTOGRAM_NORMALIZATION,
            "clahe": cls.CLAHE,
            "luminance": cls.LUMINANCE_NORMALIZATION,
            "luminance_normalization": cls.LUMINANCE_NORMALIZATION,
            "temporal": cls.TEMPORAL_SMOOTHING,
            "temporal_smoothing": cls.TEMPORAL_SMOOTHING,
            "gamma": cls.GAMMA_CORRECTION,
            "gamma_correction": cls.GAMMA_CORRECTION,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"Unsupported stabilization technique '{value}'. Allowed values: {allowed}.") from exc


@dataclass(frozen=True, slots=True)
class FrameColorStatistics:
    """Compact per-frame statistics used to normalize illumination and color."""

    luminance_mean: float
    luminance_std: float
    chroma_means: tuple[float, ...] = ()

    def blend(self, other: "FrameColorStatistics", previous_weight: float) -> "FrameColorStatistics":
        current_weight = 1.0 - previous_weight
        chroma_count = min(len(self.chroma_means), len(other.chroma_means))
        return FrameColorStatistics(
            luminance_mean=(previous_weight * self.luminance_mean) + (current_weight * other.luminance_mean),
            luminance_std=(previous_weight * self.luminance_std) + (current_weight * other.luminance_std),
            chroma_means=tuple(
                (previous_weight * self.chroma_means[index]) + (current_weight * other.chroma_means[index]) for index in range(chroma_count)
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "luminance_mean": self.luminance_mean,
            "luminance_std": self.luminance_std,
            "chroma_means": list(self.chroma_means),
        }


class _TemporalReference:
    """Maintain the target color statistics used to remove frame-to-frame flicker."""

    def __init__(self, alpha: float, scene_change_luminance_threshold: float | None) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("temporal_alpha must be in [0, 1].")
        if scene_change_luminance_threshold is not None and scene_change_luminance_threshold <= 0:
            raise ValueError("scene_change_luminance_threshold must be positive when provided.")
        self._alpha = alpha
        self._scene_change_luminance_threshold = scene_change_luminance_threshold
        self._stats: FrameColorStatistics | None = None
        self._shape: tuple[int, ...] | None = None

    def target_for(
        self,
        current: FrameColorStatistics,
        image_shape: tuple[int, ...],
        use_temporal_smoothing: bool,
    ) -> FrameColorStatistics:
        if self._should_reset(current, image_shape):
            self._stats = current
            self._shape = image_shape
            return current

        if use_temporal_smoothing:
            self._stats = self._stats.blend(current, self._alpha)

        return self._stats

    def _should_reset(self, current: FrameColorStatistics, image_shape: tuple[int, ...]) -> bool:
        if self._stats is None or self._shape != image_shape:
            return True
        if self._scene_change_luminance_threshold is None:
            return False
        return abs(current.luminance_mean - self._stats.luminance_mean) > self._scene_change_luminance_threshold


class _ColorSpaceAdapter:
    """Convert OpenCV BGR frames into the configured processing color space."""

    def __init__(self, color_space: ColorSpace) -> None:
        self.color_space = color_space

    @property
    def luminance_channel_index(self) -> int | None:
        if self.color_space == ColorSpace.HSV:
            return 2
        if self.color_space in {ColorSpace.LAB, ColorSpace.YCRCB}:
            return 0
        return None

    @property
    def chroma_channel_indices(self) -> tuple[int, ...]:
        if self.color_space == ColorSpace.RGB:
            return (0, 1, 2)
        if self.color_space == ColorSpace.HSV:
            return (1,)
        if self.color_space in {ColorSpace.LAB, ColorSpace.YCRCB}:
            return (1, 2)
        return ()

    def to_working(self, image: np.ndarray) -> np.ndarray:
        if _is_grayscale(image):
            return image.copy()
        if self.color_space == ColorSpace.RGB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.color_space == ColorSpace.HSV:
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.color_space == ColorSpace.LAB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        if self.color_space == ColorSpace.YCRCB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        raise ValueError(f"Unsupported color space: {self.color_space}")

    def from_working(self, image: np.ndarray) -> np.ndarray:
        if _is_grayscale(image):
            return image.copy()
        if self.color_space == ColorSpace.RGB:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.color_space == ColorSpace.HSV:
            return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        if self.color_space == ColorSpace.LAB:
            return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        if self.color_space == ColorSpace.YCRCB:
            return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        raise ValueError(f"Unsupported color space: {self.color_space}")

    def luminance(self, image: np.ndarray) -> np.ndarray:
        if _is_grayscale(image):
            return image
        if self.luminance_channel_index is not None:
            return image[:, :, self.luminance_channel_index]
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def apply_luminance_shift(self, image: np.ndarray, delta: float) -> np.ndarray:
        if _is_grayscale(image):
            return _clip_uint8(image.astype(np.float32) + delta)
        if self.luminance_channel_index is None:
            return _clip_uint8(image.astype(np.float32) + delta)

        output = image.copy()
        channel_index = self.luminance_channel_index
        output[:, :, channel_index] = _clip_uint8(output[:, :, channel_index].astype(np.float32) + delta)
        return output

    def apply_luminance_affine(
        self,
        image: np.ndarray,
        source_mean: float,
        target_mean: float,
        scale: float,
    ) -> np.ndarray:
        if _is_grayscale(image):
            adjusted = ((image.astype(np.float32) - source_mean) * scale) + target_mean
            return _clip_uint8(adjusted)
        if self.luminance_channel_index is None:
            adjusted = ((image.astype(np.float32) - source_mean) * scale) + target_mean
            return _clip_uint8(adjusted)

        output = image.copy()
        channel_index = self.luminance_channel_index
        channel = output[:, :, channel_index].astype(np.float32)
        output[:, :, channel_index] = _clip_uint8(((channel - source_mean) * scale) + target_mean)
        return output

    def apply_luminance_lut(self, image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        if _is_grayscale(image):
            return cv2.LUT(image, lut)
        if self.luminance_channel_index is None:
            return cv2.LUT(image, lut)

        output = image.copy()
        channel_index = self.luminance_channel_index
        output[:, :, channel_index] = cv2.LUT(output[:, :, channel_index], lut)
        return output

    def apply_luminance_channel(self, image: np.ndarray, luminance: np.ndarray) -> np.ndarray:
        if _is_grayscale(image):
            return luminance
        if self.luminance_channel_index is not None:
            output = image.copy()
            output[:, :, self.luminance_channel_index] = luminance
            return output

        original_luminance = self.luminance(image).astype(np.float32)
        ratio = (luminance.astype(np.float32) + _EPSILON) / (original_luminance + _EPSILON)
        return _clip_uint8(image.astype(np.float32) * ratio[:, :, None])


class _StatisticsExtractor:
    """Calculate color statistics without leaking color-space details to operations."""

    def measure(self, image: np.ndarray, adapter: _ColorSpaceAdapter) -> FrameColorStatistics:
        luminance = adapter.luminance(image).astype(np.float32)
        chroma_means = ()
        if not _is_grayscale(image):
            chroma_means = tuple(float(np.mean(image[:, :, index])) for index in adapter.chroma_channel_indices)
        return FrameColorStatistics(
            luminance_mean=float(np.mean(luminance)),
            luminance_std=float(np.std(luminance)),
            chroma_means=chroma_means,
        )


class _LuminanceNormalizer:
    """Normalize global luminance while capping aggressive frame shifts."""

    def __init__(self, max_shift: float) -> None:
        if max_shift <= 0:
            raise ValueError("luminance_max_shift must be positive.")
        self._max_shift = max_shift

    def apply(
        self,
        image: np.ndarray,
        adapter: _ColorSpaceAdapter,
        current: FrameColorStatistics,
        target: FrameColorStatistics,
    ) -> np.ndarray:
        delta = float(np.clip(target.luminance_mean - current.luminance_mean, -self._max_shift, self._max_shift))
        return adapter.apply_luminance_shift(image, delta)


class _HistogramNormalizer:
    """Match luminance mean and contrast to the temporal reference histogram moments."""

    def __init__(self, min_std: float, max_gain: float) -> None:
        if min_std <= 0:
            raise ValueError("histogram_min_std must be positive.")
        if max_gain < 1.0:
            raise ValueError("histogram_max_gain must be greater than or equal to 1.")
        self._min_std = min_std
        self._max_gain = max_gain

    def apply(
        self,
        image: np.ndarray,
        adapter: _ColorSpaceAdapter,
        current: FrameColorStatistics,
        target: FrameColorStatistics,
    ) -> np.ndarray:
        source_std = max(current.luminance_std, self._min_std)
        target_std = max(target.luminance_std, self._min_std)
        scale = float(np.clip(target_std / source_std, 1.0 / self._max_gain, self._max_gain))
        return adapter.apply_luminance_affine(
            image=image,
            source_mean=current.luminance_mean,
            target_mean=target.luminance_mean,
            scale=scale,
        )


class _ClaheEnhancer:
    """Apply CLAHE to luminance with a blend factor to protect fine detail."""

    def __init__(self, clip_limit: float, tile_grid_size: tuple[int, int], strength: float) -> None:
        if clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive.")
        if tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
            raise ValueError("clahe_tile_grid_size must contain positive values.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("clahe_strength must be in [0, 1].")
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self._strength = strength

    def apply(self, image: np.ndarray, adapter: _ColorSpaceAdapter) -> np.ndarray:
        luminance = adapter.luminance(image)
        enhanced = self._clahe.apply(luminance)
        blended = cv2.addWeighted(luminance, 1.0 - self._strength, enhanced, self._strength, 0)
        return adapter.apply_luminance_channel(image, blended)


class _GammaCorrector:
    """Apply bounded gamma correction toward the target luminance level."""

    def __init__(self, gamma: float | None, gamma_limits: tuple[float, float]) -> None:
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be positive when provided.")
        lower, upper = gamma_limits
        if lower <= 0 or upper <= 0 or lower > upper:
            raise ValueError("gamma_limits must contain positive values ordered as (lower, upper).")
        self._gamma = gamma
        self._gamma_limits = gamma_limits

    def apply(
        self,
        image: np.ndarray,
        adapter: _ColorSpaceAdapter,
        current: FrameColorStatistics,
        target: FrameColorStatistics,
    ) -> np.ndarray:
        gamma = self._gamma if self._gamma is not None else self._estimated_gamma(current, target)
        gamma = float(np.clip(gamma, self._gamma_limits[0], self._gamma_limits[1]))
        values = np.arange(256, dtype=np.float32) / _MAX_CHANNEL_VALUE
        lut = _clip_uint8((values**gamma) * _MAX_CHANNEL_VALUE)
        return adapter.apply_luminance_lut(image, lut)

    @staticmethod
    def _estimated_gamma(current: FrameColorStatistics, target: FrameColorStatistics) -> float:
        current_mean = float(np.clip(current.luminance_mean / _MAX_CHANNEL_VALUE, 0.02, 0.98))
        target_mean = float(np.clip(target.luminance_mean / _MAX_CHANNEL_VALUE, 0.02, 0.98))
        return float(np.log(target_mean) / np.log(current_mean))


class _ChromaNormalizer:
    """Reduce color-cast drift by softly aligning chroma channel means."""

    def __init__(self, strength: float, max_shift: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("chroma_strength must be in [0, 1].")
        if max_shift <= 0:
            raise ValueError("max_chroma_shift must be positive.")
        self._strength = strength
        self._max_shift = max_shift

    def apply(
        self,
        image: np.ndarray,
        adapter: _ColorSpaceAdapter,
        target: FrameColorStatistics,
        statistics_extractor: _StatisticsExtractor,
    ) -> np.ndarray:
        if _is_grayscale(image) or self._strength == 0.0:
            return image

        chroma_indices = adapter.chroma_channel_indices
        if not chroma_indices or len(target.chroma_means) != len(chroma_indices):
            return image

        current = statistics_extractor.measure(image, adapter)
        output = image.copy()
        for position, channel_index in enumerate(chroma_indices):
            delta = (target.chroma_means[position] - current.chroma_means[position]) * self._strength
            delta = float(np.clip(delta, -self._max_shift, self._max_shift))
            output[:, :, channel_index] = _clip_uint8(output[:, :, channel_index].astype(np.float32) + delta)
        return output


class _ArtifactBuilder:
    """Build optional metadata artifacts without coupling the cleaner to UI classes."""

    def __init__(self, emit_comparison_overlay: bool, emit_intermediate_artifacts: bool, emit_metrics: bool) -> None:
        self._emit_comparison_overlay = emit_comparison_overlay
        self._emit_intermediate_artifacts = emit_intermediate_artifacts
        self._emit_metrics = emit_metrics

    def build(
        self,
        original_image: np.ndarray,
        cleaned_image: np.ndarray,
        original_working: np.ndarray,
        processed_working: np.ndarray,
        original_stats: FrameColorStatistics,
        cleaned_stats: FrameColorStatistics,
        target_stats: FrameColorStatistics,
        color_space: ColorSpace,
        techniques: Sequence[StabilizationTechnique],
        adapter: _ColorSpaceAdapter,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self._emit_metrics:
            payload["metrics"] = {
                "color_space": color_space.value,
                "techniques": [technique.value for technique in techniques],
                "original": original_stats.as_dict(),
                "cleaned": cleaned_stats.as_dict(),
                "target": target_stats.as_dict(),
            }
        if self._emit_comparison_overlay:
            payload["comparison_overlay"] = self._comparison_overlay(original_image, cleaned_image)
        if self._emit_intermediate_artifacts:
            payload["intermediate_artifacts"] = {
                "working_before": original_working.copy(),
                "working_after": processed_working.copy(),
                "luminance_before": adapter.luminance(original_working).copy(),
                "luminance_after": adapter.luminance(processed_working).copy(),
            }
        return payload

    @staticmethod
    def _comparison_overlay(original_image: np.ndarray, cleaned_image: np.ndarray) -> np.ndarray:
        left = _to_bgr_preview(original_image)
        right = _to_bgr_preview(cleaned_image)
        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)

        overlay = cv2.hconcat([left, right])
        divider_x = left.shape[1]
        cv2.line(overlay, (divider_x, 0), (divider_x, overlay.shape[0] - 1), (255, 255, 255), 1)
        _put_label(overlay, "original", (8, 22))
        _put_label(overlay, "cleaned", (divider_x + 8, 22))
        return overlay


class ColorStabilizationFrameCleaner(IFrameCleaner):
    """
    Stabilize frame illumination and chromatic drift while preserving image detail.

    The cleaner estimates a temporal reference from frame luminance and chroma
    statistics, then applies only the selected bounded operations. OpenCV frames
    are returned in their original BGR/grayscale layout so downstream extractors
    can consume the cleaned sequence without format-specific coupling.
    """

    def __init__(
        self,
        color_space: str | ColorSpace | None = None,
        techniques: Sequence[str | StabilizationTechnique] | None = None,
        stabilization_strength: float | None = None,
        temporal_alpha: float | None = None,
        histogram_min_std: float | None = None,
        histogram_max_gain: float | None = None,
        luminance_max_shift: float | None = None,
        gamma: float | None = None,
        gamma_limits: tuple[float, float] | None = None,
        clahe_clip_limit: float | None = None,
        clahe_tile_grid_size: tuple[int, int] | Sequence[int] | None = None,
        clahe_strength: float | None = None,
        stabilize_chroma: bool | None = None,
        chroma_strength: float | None = None,
        max_chroma_shift: float | None = None,
        scene_change_luminance_threshold: float | None = None,
        emit_metrics: bool | None = None,
        emit_comparison_overlay: bool | None = None,
        emit_intermediate_artifacts: bool | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.color_space = ColorSpace.from_value(self._option("color_space", color_space, ColorSpace.LAB.value))
        self.techniques = self._parse_techniques(self._option("techniques", techniques, _DEFAULT_TECHNIQUES))
        self.stabilization_strength = self._bounded_float(
            "stabilization_strength",
            stabilization_strength,
            default=0.85,
            lower=0.0,
            upper=1.0,
        )
        self._adapter = _ColorSpaceAdapter(self.color_space)
        self._statistics_extractor = _StatisticsExtractor()
        self._temporal_reference = _TemporalReference(
            alpha=self._bounded_float("temporal_alpha", temporal_alpha, default=0.92, lower=0.0, upper=1.0),
            scene_change_luminance_threshold=self._optional_positive_float(
                "scene_change_luminance_threshold",
                scene_change_luminance_threshold,
            ),
        )
        self._luminance_normalizer = _LuminanceNormalizer(max_shift=self._positive_float("luminance_max_shift", luminance_max_shift, default=48.0))
        self._histogram_normalizer = _HistogramNormalizer(
            min_std=self._positive_float("histogram_min_std", histogram_min_std, default=4.0),
            max_gain=self._positive_float("histogram_max_gain", histogram_max_gain, default=1.35),
        )
        self._clahe_enhancer = _ClaheEnhancer(
            clip_limit=self._positive_float("clahe_clip_limit", clahe_clip_limit, default=2.0),
            tile_grid_size=self._tile_grid_size(clahe_tile_grid_size),
            strength=self._bounded_float("clahe_strength", clahe_strength, default=0.35, lower=0.0, upper=1.0),
        )
        self._gamma_corrector = _GammaCorrector(
            gamma=self._optional_positive_float("gamma", gamma),
            gamma_limits=self._gamma_limits(gamma_limits),
        )
        self._stabilize_chroma = self._bool_option("stabilize_chroma", stabilize_chroma, default=True)
        self._chroma_normalizer = _ChromaNormalizer(
            strength=self._bounded_float("chroma_strength", chroma_strength, default=0.20, lower=0.0, upper=1.0),
            max_shift=self._positive_float("max_chroma_shift", max_chroma_shift, default=18.0),
        )
        self._emit_comparison_overlay = self._bool_option("emit_comparison_overlay", emit_comparison_overlay, default=False)
        self._emit_intermediate_artifacts = self._bool_option(
            "emit_intermediate_artifacts",
            emit_intermediate_artifacts,
            default=False,
        )
        self._emit_metrics = self._bool_option("emit_metrics", emit_metrics, default=True)
        self._artifact_builder = _ArtifactBuilder(
            emit_comparison_overlay=self._emit_comparison_overlay,
            emit_intermediate_artifacts=self._emit_intermediate_artifacts,
            emit_metrics=self._emit_metrics,
        )

    def clean(self, frame: Frame) -> Frame:
        image = self._validated_image(frame.frame)
        working = self._adapter.to_working(image)
        original_working = working.copy()
        original_stats = self._statistics_extractor.measure(working, self._adapter)
        target_stats = self._temporal_reference.target_for(
            current=original_stats,
            image_shape=image.shape,
            use_temporal_smoothing=StabilizationTechnique.TEMPORAL_SMOOTHING in self.techniques,
        )

        for technique in self.techniques:
            working = self._apply_technique(technique, working, target_stats)

        if self._stabilize_chroma:
            working = self._chroma_normalizer.apply(
                image=working,
                adapter=self._adapter,
                target=target_stats,
                statistics_extractor=self._statistics_extractor,
            )

        processed_image = self._adapter.from_working(working)
        cleaned_image = self._blend_with_original(image, processed_image)
        cleaned_working = self._adapter.to_working(cleaned_image)
        cleaned_stats = self._statistics_extractor.measure(cleaned_working, self._adapter)
        metadata = dict(frame.metadata)
        metadata[_METADATA_KEY] = self._artifact_builder.build(
            original_image=image,
            cleaned_image=cleaned_image,
            original_working=original_working,
            processed_working=working,
            original_stats=original_stats,
            cleaned_stats=cleaned_stats,
            target_stats=target_stats,
            color_space=self.color_space,
            techniques=self.techniques,
            adapter=self._adapter,
        )

        return Frame(
            image=cleaned_image,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata=metadata,
        )

    def emit_intermediate_artifacts(
        self,
        original_frame: Frame,
        cleaned_frame: Frame,
        context: Any,
    ) -> tuple[Any, ...]:
        """
        Emit optional pipeline-level intermediate artifacts when the host stage supports them.

        The import is intentionally dynamic: older pipeline versions only know the
        IFrameCleaner.clean contract and can still use this cleaner without the
        intermediate-artifact classes being present.
        """
        if not self._emit_intermediate_artifacts and not self._emit_comparison_overlay:
            return ()

        try:
            from library.core.artifacts.MaskArtifacts import IntermediateFrameArtifact
        except Exception:
            return ()

        stabilization_payload = dict(cleaned_frame.metadata.get(_METADATA_KEY, {}))
        metrics = dict(stabilization_payload.get("metrics", {}))
        artifact_image = stabilization_payload.get("comparison_overlay") if self._emit_comparison_overlay else None
        if artifact_image is None:
            artifact_image = cleaned_frame.image

        metadata = {
            "cleaner_name": type(self).__name__,
            "color_space": self.color_space.value,
            "techniques": [technique.value for technique in self.techniques],
            "metrics": metrics,
        }
        return (
            IntermediateFrameArtifact(
                image=artifact_image,
                stage_name=getattr(context, "stage_name", type(self).__name__),
                frame_index=cleaned_frame.index,
                timestamp_seconds=cleaned_frame.timestamp_seconds,
                color_space=self.color_space.value,
                original_image=original_frame.image,
                cleaned_image=cleaned_frame.image,
                stage_metadata=metadata,
                metadata=metadata,
                config=self._resolved_config(),
            ),
        )

    def _apply_technique(
        self,
        technique: StabilizationTechnique,
        working: np.ndarray,
        target_stats: FrameColorStatistics,
    ) -> np.ndarray:
        if technique == StabilizationTechnique.TEMPORAL_SMOOTHING:
            return working

        current_stats = self._statistics_extractor.measure(working, self._adapter)
        if technique == StabilizationTechnique.LUMINANCE_NORMALIZATION:
            return self._luminance_normalizer.apply(working, self._adapter, current_stats, target_stats)
        if technique == StabilizationTechnique.HISTOGRAM_NORMALIZATION:
            return self._histogram_normalizer.apply(working, self._adapter, current_stats, target_stats)
        if technique == StabilizationTechnique.CLAHE:
            return self._clahe_enhancer.apply(working, self._adapter)
        if technique == StabilizationTechnique.GAMMA_CORRECTION:
            return self._gamma_corrector.apply(working, self._adapter, current_stats, target_stats)
        raise ValueError(f"Unsupported stabilization technique: {technique}")

    def _blend_with_original(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        if self.stabilization_strength >= 1.0:
            return processed
        if self.stabilization_strength <= 0.0:
            return original.copy()
        return cv2.addWeighted(
            original,
            1.0 - self.stabilization_strength,
            processed,
            self.stabilization_strength,
            0,
        )

    def _option(self, key: str, explicit_value: Any, default: Any) -> Any:
        return explicit_value if explicit_value is not None else self.config.get(key, default)

    def _resolved_config(self) -> dict[str, Any]:
        return {
            **dict(self.config),
            "color_space": self.color_space.value,
            "techniques": [technique.value for technique in self.techniques],
            "stabilization_strength": self.stabilization_strength,
            "stabilize_chroma": self._stabilize_chroma,
            "emit_metrics": self._emit_metrics,
            "emit_comparison_overlay": self._emit_comparison_overlay,
            "emit_intermediate_artifacts": self._emit_intermediate_artifacts,
        }

    def _positive_float(self, key: str, explicit_value: float | None, default: float) -> float:
        value = float(self._option(key, explicit_value, default))
        if value <= 0:
            raise ValueError(f"{key} must be positive.")
        return value

    def _optional_positive_float(self, key: str, explicit_value: float | None) -> float | None:
        value = self._option(key, explicit_value, None)
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            raise ValueError(f"{key} must be positive when provided.")
        return value

    def _bounded_float(
        self,
        key: str,
        explicit_value: float | None,
        default: float,
        lower: float,
        upper: float,
    ) -> float:
        value = float(self._option(key, explicit_value, default))
        if not lower <= value <= upper:
            raise ValueError(f"{key} must be in [{lower}, {upper}].")
        return value

    def _bool_option(self, key: str, explicit_value: bool | None, default: bool) -> bool:
        return bool(self._option(key, explicit_value, default))

    def _tile_grid_size(self, explicit_value: tuple[int, int] | Sequence[int] | None) -> tuple[int, int]:
        raw_value = self._option("clahe_tile_grid_size", explicit_value, (8, 8))
        values = tuple(int(value) for value in raw_value)
        if len(values) != 2:
            raise ValueError("clahe_tile_grid_size must contain exactly two integers.")
        if values[0] <= 0 or values[1] <= 0:
            raise ValueError("clahe_tile_grid_size must contain positive integers.")
        return values

    def _gamma_limits(self, explicit_value: tuple[float, float] | None) -> tuple[float, float]:
        raw_value = self._option("gamma_limits", explicit_value, (0.75, 1.35))
        values = tuple(float(value) for value in raw_value)
        if len(values) != 2:
            raise ValueError("gamma_limits must contain exactly two floats.")
        return values

    @staticmethod
    def _parse_techniques(raw_techniques: Sequence[str | StabilizationTechnique]) -> tuple[StabilizationTechnique, ...]:
        techniques = tuple(StabilizationTechnique.from_value(technique) for technique in raw_techniques)
        if not techniques:
            raise ValueError("At least one stabilization technique must be enabled.")
        return techniques

    @staticmethod
    def _validated_image(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("ColorStabilizationFrameCleaner expects frame.image to be a numpy.ndarray.")
        if image.size == 0:
            raise ValueError("ColorStabilizationFrameCleaner cannot process an empty frame.")
        if image.dtype != np.uint8:
            raise ValueError("ColorStabilizationFrameCleaner expects uint8 frames.")
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError("ColorStabilizationFrameCleaner expects grayscale or 3-channel BGR frames.")


def _is_grayscale(image: np.ndarray) -> bool:
    return image.ndim == 2


def _clip_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0, _MAX_CHANNEL_VALUE).astype(np.uint8)


def _to_bgr_preview(image: np.ndarray) -> np.ndarray:
    if _is_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
