from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Protocol

import numpy as np
import numpy.typing as npt

from library.core.artifacts.Frame import Frame
from library.frame_processors.dynamic_object_removal.config import DynamicObjectRemovalConfig

log = logging.getLogger(__name__)


class BackgroundEstimator(Protocol):
    """Estimate a static background image from an ordered frame sequence."""

    def estimate(self, frames: Sequence[Frame]) -> npt.NDArray:
        """Return an image with the same shape and dtype as the input frames."""


class TemporalMedianBackgroundEstimator:
    """Estimate a static background by taking the temporal median over sampled frames."""

    def __init__(self, config: DynamicObjectRemovalConfig) -> None:
        self._config = config
        self.last_sampled_indices: tuple[int, ...] = ()

    def estimate(self, frames: Sequence[Frame]) -> npt.NDArray:
        if not frames:
            raise ValueError("Cannot estimate background from an empty frame sequence.")

        sample_indices = self._sample_indices(len(frames))
        sample_bytes = sum(int(frames[index].image.nbytes) for index in sample_indices)
        log.info(
            "Estimating temporal median background from %s/%s frames (%.1f MiB sample stack).",
            len(sample_indices),
            len(frames),
            sample_bytes / (1024 * 1024),
        )
        if sample_bytes > self._config.max_sample_stack_bytes:
            raise ValueError(
                "Dynamic object removal sample stack would exceed "
                f"{self._config.max_sample_stack_bytes} bytes. Reduce max_sampled_frames, resize the video, "
                "or increase max_sample_stack_bytes explicitly."
            )

        samples = np.stack([frames[index].image for index in sample_indices], axis=0)
        self.last_sampled_indices = sample_indices
        background = self._median(samples).astype(frames[0].image.dtype, copy=False)
        log.info("Temporal median background estimated.")
        return background

    def _sample_indices(self, frame_count: int) -> tuple[int, ...]:
        if frame_count <= 0:
            return ()
        if frame_count <= self._config.max_sampled_frames:
            return tuple(range(frame_count))

        candidate_indices = tuple(range(0, frame_count, self._config.sampling_stride))
        if not candidate_indices:
            return (0,)
        if len(candidate_indices) <= self._config.max_sampled_frames:
            return candidate_indices

        positions = np.linspace(
            0,
            len(candidate_indices) - 1,
            num=self._config.max_sampled_frames,
            dtype=np.int64,
        )
        return tuple(candidate_indices[int(position)] for position in positions)

    @staticmethod
    def _median(samples: npt.NDArray) -> npt.NDArray:
        """
        Return temporal median with reduced temporary memory for uint8 video samples.

        ``np.median`` promotes uint8 arrays to float64, which is expensive for
        full-frame video. Partition keeps the sample stack in its native dtype.
        """
        if not np.issubdtype(samples.dtype, np.integer):
            return np.median(samples, axis=0)

        sample_count = int(samples.shape[0])
        middle = sample_count // 2
        if sample_count % 2 == 1:
            return np.partition(samples, middle, axis=0)[middle]

        partitioned = np.partition(samples, (middle - 1, middle), axis=0)
        if np.issubdtype(samples.dtype, np.signedinteger):
            accumulator_dtype = np.int64
        else:
            accumulator_dtype = np.uint32 if samples.dtype.itemsize <= 2 else np.uint64
        lower = partitioned[middle - 1].astype(accumulator_dtype)
        upper = partitioned[middle].astype(accumulator_dtype)
        return ((lower + upper) // 2).astype(samples.dtype)
