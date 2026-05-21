from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from library.core.artifacts.COCOPoseFrameData import COCOPoseFrameData, COCOPoseSequenceData
from library.core.artifacts.COCOSkeletonSignalSample import COCOSkeletonSignalSample
from library.core.artifacts.DataBuffer import DataBuffer
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

    def analyze(self, signal: ISignal) -> IData:
        output = self._default_buffer or DataBuffer()
        return self.analyze_into(signal, output)

    def analyze_into(
        self,
        signal: Iterable[ISignalSample],
        output_buffer: DataBuffer,
    ) -> COCOPoseSequenceData:
        frames: list[COCOPoseFrameData] = []
        try:
            for sample in signal:
                pose_frame = self._map_sample(sample)
                frames.append(pose_frame)
                output_buffer.put(pose_frame)
        finally:
            output_buffer.close()

        return COCOPoseSequenceData(
            frames=frames,
            metadata={"frames": len(frames)},
        )

    @staticmethod
    def _map_sample(sample: ISignalSample) -> COCOPoseFrameData:
        if not isinstance(sample, COCOSkeletonSignalSample):
            raise TypeError(
                "COCOPoseStreamAnalyzer requires COCOSkeletonSignalSample, "
                f"got {type(sample).__name__}."
            )

        return COCOPoseFrameData(
            frame_index=sample.frame_index,
            skeleton=sample.skeleton,
            confidence=sample.confidence,
            centroid=sample.centroid,
            timestamp_seconds=sample.timestamp_seconds,
            frame_size=sample.metadata.get("frame_size"),
            metadata=dict(sample.metadata),
        )
