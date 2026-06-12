from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from sef.core.artifacts.signal_sample.COCOSkeletonSignalSample import COCOSkeletonSignalSample
from sef.core.artifacts.Signal import Signal
from sef.core.interfaces.BufferContracts import IBuffer
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalSample import ISignalSample
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.interfaces.StreamingContracts import IStreamingSignalCleaner
from sef.core.pose.COCOSkeletonNormalizer import (
    COCOSkeletonNormalizationConfig,
    COCOSkeletonNormalizer,
)


class COCOSkeletonNormalizationSignalCleaner(IStreamingSignalCleaner):
    """
    Streaming skeleton normalization cleaner.

    Operations:
        - pelvis centering
        - torso scale normalization
        - optional shoulder rotation alignment
    """

    capabilities = StageCapabilities.streaming(
        stateful=False,
        preserves_order=True,
        realtime_safe=True,
    )

    def __init__(
        self,
        *,
        center_on_pelvis: bool = True,
        normalize_scale: bool = True,
        align_rotation: bool = False,
        min_scale: float = 1e-6,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self._normalizer = COCOSkeletonNormalizer(
            COCOSkeletonNormalizationConfig(
                center_on_pelvis=bool(center_on_pelvis),
                normalize_scale=bool(normalize_scale),
                align_rotation=bool(align_rotation),
                min_scale=float(min_scale),
            )
        )

    def clean(self, signal: ISignal) -> ISignal:
        samples = list(signal)

        if any(not isinstance(sample, COCOSkeletonSignalSample) for sample in samples):
            raise TypeError(
                "COCOSkeletonNormalizationCleaner requires "
                "COCOSkeletonSignalSample inputs."
            )

        cleaned_samples = [self._clean_sample(sample) for sample in samples]

        return Signal(
            cleaned_samples,
            config=dict(signal.config),
        )

    def clean_into(
        self,
        input_signal: Iterable[ISignalSample],
        output_buffer: IBuffer[ISignalSample],
    ) -> None:
        try:
            for sample in input_signal:

                if not isinstance(sample, COCOSkeletonSignalSample):
                    raise TypeError(
                        "COCOSkeletonNormalizationCleaner requires "
                        "COCOSkeletonSignalSample inputs."
                    )

                output_buffer.put(self._clean_sample(sample))

        finally:
            output_buffer.close()

    def _clean_sample(
        self,
        sample: COCOSkeletonSignalSample,
    ) -> COCOSkeletonSignalSample:
        normalization = self._normalizer.normalize(sample.skeleton)
        skeleton = normalization.skeleton
        centroid = COCOSkeletonNormalizer.compute_centroid(skeleton)

        return COCOSkeletonSignalSample(
            frame_index=sample.frame_index,
            skeleton=skeleton,
            confidence=np.asarray(sample.confidence).copy(),
            centroid=centroid,
            timestamp_seconds=sample.timestamp_seconds,
            metadata={
                **dict(sample.metadata),
                "skeleton_normalization": normalization.metadata(),
            },
        )
