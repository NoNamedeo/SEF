from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np

from library.core.artifacts.data.ArucoDisplacementData import ArucoMarkerDisplacementData
from library.core.artifacts.signal_sample.ArucoMarkerSignalSample import ArucoMarkerObservation
from library.core.interfaces.IData import IData
from library.core.interfaces.StageCapabilities import StageCapabilities
from library.core.interfaces.StreamingContracts import IStreamingVisualizer
from library.core.visualization.VisualArtifact import VideoLikeArtifact
from library.core.visualization.VisualizationContext import VisualizationContext
from library.visualizers.TrackingVideoVisualizer import TrackingVideoVisualizer


class ArucoAnnotatedVideoVisualizer(TrackingVideoVisualizer, IStreamingVisualizer):
    """Return lazy annotated MP4 artifacts for ArUco marker displacement data."""

    capabilities = StageCapabilities.streaming(
        stateful=True,
        preserves_order=True,
        realtime_safe=False,
    )

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VideoLikeArtifact, ...]:
        if not isinstance(data, ArucoMarkerDisplacementData):
            raise TypeError(
                "ArucoAnnotatedVideoVisualizer requires ArucoMarkerDisplacementData, "
                f"got {type(data).__name__}."
            )

        return self._build_video_artifact(data, context)

    def render_stream(
        self,
        data: Iterable[IData],
        context: VisualizationContext | None = None,
    ) -> tuple[VideoLikeArtifact, ...]:
        displacement_data = ArucoMarkerDisplacementData.from_stream_items(data)
        if not displacement_data.frames:
            return ()
        return self._build_video_artifact(displacement_data, context)

    def _artifact_description(self) -> str:
        return "Annotated video with ArUco corners, centers, and marker ids."

    def _annotation_context(
        self,
        data: ArucoMarkerDisplacementData,
    ) -> dict[int, dict[int, tuple[float, float, float]]]:
        return self._displacement_lookup(data)

    def _draw_annotation_frame(
        self,
        image: np.ndarray,
        annotation_frame: Any,
        data: Any,
        annotation_context: Any,
    ) -> np.ndarray:
        return self._draw_markers(
            image=image,
            observations=annotation_frame.markers,
            displacement_by_marker=annotation_context.get(annotation_frame.frame_index, {}),
        )

    def _draw_markers(
        self,
        *,
        image: np.ndarray,
        observations: list[ArucoMarkerObservation],
        displacement_by_marker: dict[int, tuple[float, float, float]],
    ) -> np.ndarray:
        annotated = image.copy()
        for observation in observations:
            if not observation.detected or observation.corners is None or observation.center is None:
                continue

            color = self._track_color(observation.marker_id)
            points = [
                (int(round(point[0])), int(round(point[1])))
                for point in observation.corners
            ]
            for start_point, end_point in zip(points, points[1:] + points[:1]):
                cv2.line(annotated, start_point, end_point, color, 2)

            center_x, center_y = observation.center
            cv2.circle(annotated, (int(round(center_x)), int(round(center_y))), 4, color, -1)

            label = f"ID {observation.marker_id}"
            displacement = displacement_by_marker.get(observation.marker_id)
            if displacement is not None:
                dx, dy, magnitude = displacement
                label = f"{label} | dx={dx:.2f} dy={dy:.2f} | |d|={magnitude:.2f}"

            cv2.putText(
                annotated,
                label,
                (points[0][0], max(18, points[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return annotated

    @staticmethod
    def _displacement_lookup(
        data: ArucoMarkerDisplacementData,
    ) -> dict[int, dict[int, tuple[float, float, float]]]:
        lookup: dict[int, dict[int, tuple[float, float, float]]] = {}
        for series in data.series:
            for frame_index, dx, dy, magnitude in zip(
                series.frame_indices,
                series.displacement_x,
                series.displacement_y,
                series.displacement_magnitude,
            ):
                if not all(math.isfinite(value) for value in (dx, dy, magnitude)):
                    continue
                lookup.setdefault(frame_index, {})[series.marker_id] = (dx, dy, magnitude)
        return lookup
