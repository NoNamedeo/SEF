from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from sef.builtin.visualizers.COCO_pose.COCOPoseFrameRenderer import COCOPoseFrameRenderer, COCOPoseRenderConfig
from sef.core.artifacts.data.COCOPoseFrameData import COCOPoseFrameData, COCOPoseSequenceData
from sef.core.artifacts.data.COCOPoseTennisFrameData import COCOPoseTennisFrameData, COCOPoseTennisSequenceData
from sef.core.interfaces.IData import IData
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.interfaces.StreamingContracts import IStreamingVisualizer
from sef.core.realtime.IRealtimeFrameSink import IRealtimeFrameSink
from sef.core.realtime.NullRealtimeFrameSink import NullRealtimeFrameSink
from sef.core.realtime.RealtimeFrame import RealtimeFrame
from sef.core.visualization.VisualArtifact import VisualArtifact
from sef.core.visualization.VisualizationContext import VisualizationContext


class RealtimeCOCOPoseFrameVisualizer(IStreamingVisualizer):
    """
    Stream COCO pose frames to an injected realtime sink.

    The class is UI-framework agnostic: OpenCV is used only for drawing pixels,
    while delivery is delegated to IRealtimeFrameSink. This makes the visualizer
    reusable for Streamlit, websocket dashboards, recording adapters, or tests.
    """

    capabilities = StageCapabilities.streaming(
        stateful=True,
        preserves_order=True,
        realtime_safe=True,
    )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        sink: IRealtimeFrameSink | None = None,
        renderer: COCOPoseFrameRenderer | None = None,
    ) -> None:
        super().__init__(config)
        self._sink = sink or NullRealtimeFrameSink()
        self._renderer = renderer or COCOPoseFrameRenderer(COCOPoseRenderConfig.from_mapping(self.config))
        self._min_publish_interval_seconds = max(0.0, float(self.config.get("min_publish_interval_ms", 0)) / 1000.0)
        self._publish_every_n_frames = max(1, int(self.config.get("publish_every_n_frames", 1)))

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        if isinstance(data, COCOPoseSequenceData | COCOPoseTennisSequenceData):
            return self.render_stream(data.frames, context)
        if isinstance(data, COCOPoseFrameData | COCOPoseTennisFrameData):
            return self.render_stream((data,), context)
        raise TypeError(
            "RealtimeCOCOPoseFrameVisualizer requires COCO pose frame/sequence data, "
            f"got {type(data).__name__}."
        )

    def render_stream(
        self,
        data: Iterable[IData],
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        last_publish_time = 0.0
        try:
            for sequence_index, item in enumerate(data):
                pose_frame = self._require_pose_frame(item)
                if sequence_index % self._publish_every_n_frames != 0:
                    continue
                now = time.monotonic()
                if self._min_publish_interval_seconds > 0 and now - last_publish_time < self._min_publish_interval_seconds:
                    continue

                canvas = self._renderer.render(pose_frame)
                self._sink.publish(
                    RealtimeFrame(
                        image=canvas,
                        color_space="BGR",
                        frame_index=int(getattr(pose_frame, "frame_index", sequence_index)),
                        timestamp_seconds=getattr(pose_frame, "timestamp_seconds", None),
                        metadata=self._metadata_for(pose_frame, context),
                    )
                )
                last_publish_time = now
        finally:
            self._sink.close()

        return ()

    @staticmethod
    def _require_pose_frame(item: IData) -> COCOPoseFrameData | COCOPoseTennisFrameData:
        if isinstance(item, COCOPoseFrameData | COCOPoseTennisFrameData):
            return item
        raise TypeError(f"Expected COCO pose frame data, got {type(item).__name__}.")

    @staticmethod
    def _metadata_for(
        pose_frame: COCOPoseFrameData | COCOPoseTennisFrameData,
        context: VisualizationContext | None,
    ) -> dict[str, Any]:
        metadata = dict(getattr(pose_frame, "metadata", {}) or {})
        if context is not None:
            metadata.update(
                {
                    "pipeline_id": context.pipeline_id,
                    "analyzer_name": context.analyzer_name,
                    "visualizer_name": context.visualizer_name,
                    "result_index": context.result_index,
                }
            )
        movement = getattr(pose_frame, "tennis_movement", None)
        if movement is not None:
            metadata["tennis_movement"] = movement
        metadata["preview_stage"] = "coco_pose_visualizer"
        metadata["preview_priority"] = 100
        return metadata
