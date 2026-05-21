from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from library.core.artifacts.COCOPoseFrameData import COCOPoseFrameData, COCOPoseSequenceData
from library.core.artifacts.COCOSkeletonSignalSample import COCOSkeletonSignalSample
from library.core.artifacts.DataBuffer import DataBuffer
from library.core.interfaces.BufferContracts import IBuffer
from library.core.interfaces.IData import IData
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalSample import ISignalSample
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import IStreamingAnalyzer


class COCOPoseStreamAnalyzer(IStreamingAnalyzer):
    """Map COCO skeleton signal samples to visualization-ready pose data."""

    capabilities = StageCapabilities.streaming(
        stateful=False,
        preserves_order=True,
        realtime_safe=True,
    )

    def __init__(
        self,
        buffer: DataBuffer | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._default_buffer = buffer
        self._retain_frames = bool(self.config.get("retain_frames", False))

    def analyze(self, signal: ISignal) -> IData:
        output = self._default_buffer or DataBuffer()
        return self.analyze_into(signal, output)

    def analyze_into(
        self,
        signal: Iterable[ISignalSample],
        output_buffer: IBuffer[IData],
    ) -> COCOPoseSequenceData:
        frames: list[COCOPoseFrameData] = []
        frame_count = 0
        try:
            signal_iterator = iter(signal)
            while not output_buffer.closed:
                sample = next(signal_iterator)
                pose_frame = self._map_sample(sample)
                frame_count += 1
                if self._retain_frames:
                    frames.append(pose_frame)
                output_buffer.put(pose_frame)
        except StopIteration:
            pass
        finally:
            if output_buffer.closed:
                self._abort_upstream(signal)
            output_buffer.close()

        return COCOPoseSequenceData(
            frames=frames,
            metadata={
                "frames": frame_count,
                "retained_frames": len(frames),
            },
        )

    @staticmethod
    def _map_sample(sample: ISignalSample) -> COCOPoseFrameData:
        if not isinstance(sample, COCOSkeletonSignalSample):
            raise TypeError(
                "COCOPoseStreamAnalyzer requires COCOSkeletonSignalSample, "
                f"got {type(sample).__name__}."
            )

        metadata = dict(sample.metadata)
        frame_image = metadata.pop("frame_image", None)

        return COCOPoseFrameData(
            frame_index=sample.frame_index,
            skeleton=sample.skeleton,
            confidence=sample.confidence,
            centroid=sample.centroid,
            timestamp_seconds=sample.timestamp_seconds,
            frame_size=metadata.get("frame_size"),
            frame_image=frame_image,
            metadata=metadata,
        )

    @staticmethod
    def _abort_upstream(signal: Iterable[ISignalSample]) -> None:
        abort = getattr(signal, "abort", None)
        if callable(abort):
            abort()
