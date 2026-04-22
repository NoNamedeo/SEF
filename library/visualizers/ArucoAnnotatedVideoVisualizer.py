from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import cv2

from library.core.artifacts.ArucoDisplacementData import ArucoMarkerDisplacementData
from library.core.artifacts.ArucoMarkerSignalSample import ArucoMarkerObservation
from library.core.visualization.VisualArtifact import VideoArtifact
from library.core.visualization.VisualizationContext import VisualizationContext
from library.visualizers.TrackingVideoVisualizer import TrackingVideoVisualizer


class ArucoAnnotatedVideoVisualizer(TrackingVideoVisualizer):
    """Render ArUco detections as an annotated MP4 artifact."""

    def render(
        self,
        data: ArucoMarkerDisplacementData,
        context: VisualizationContext | None = None,
    ) -> tuple[VideoArtifact, ...]:
        if not isinstance(data, ArucoMarkerDisplacementData):
            raise TypeError(
                "ArucoAnnotatedVideoVisualizer requires ArucoMarkerDisplacementData, "
                f"got {type(data).__name__}."
            )

        artifact_bytes, resolved_codec = self._render_video_bytes(data)
        return (
            VideoArtifact(
                kind="video",
                title=data.title,
                description="Annotated video with ArUco corners, centers, and marker ids.",
                metadata=self._artifact_metadata(
                    context,
                    extra={
                        "source_path": data.source_path,
                        "resize": data.resize,
                        "frame_count": len(data.frames),
                        "video_codec": resolved_codec,
                    },
                ),
                mime_type="video/mp4",
                data=artifact_bytes,
            ),
        )

    def _render_video_bytes(self, data: ArucoMarkerDisplacementData) -> tuple[bytes, str]:
        if not data.frames:
            raise ValueError("ArucoMarkerDisplacementData.frames cannot be empty.")
        if not data.source_path:
            raise ValueError("ArucoMarkerDisplacementData.source_path is required to render an annotated video.")

        source_path = Path(data.source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"ArUco source video not found: {source_path}")

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video for ArUco playback: {source_path}")

        ordered_frames = sorted(data.frames, key=lambda item: item.frame_index)
        output_size = self._resolve_output_size(capture, data)
        output_fps = data.fps or float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or self.DEFAULT_FPS

        file_descriptor, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(file_descriptor)
        output_path = Path(temp_path)
        writer, resolved_codec = self._open_video_writer(
            output_path=output_path,
            fps=output_fps,
            size=output_size,
        )
        if not writer.isOpened():
            capture.release()
            output_path.unlink(missing_ok=True)
            raise RuntimeError("Failed to initialize the MP4 writer for ArUco playback.")

        displacement_lookup = self._displacement_lookup(data)
        try:
            for frame_sample in ordered_frames:
                image = self._read_frame(capture, frame_sample.frame_index)
                if data.resize is not None:
                    image = cv2.resize(image, data.resize)
                annotated = self._draw_markers(
                    image=image,
                    observations=frame_sample.markers,
                    displacement_by_marker=displacement_lookup.get(frame_sample.frame_index, {}),
                )
                writer.write(annotated)
        finally:
            writer.release()
            capture.release()

        try:
            return output_path.read_bytes(), resolved_codec
        finally:
            output_path.unlink(missing_ok=True)

    @staticmethod
    def _resolve_output_size(
        capture: cv2.VideoCapture,
        data: ArucoMarkerDisplacementData,
    ) -> tuple[int, int]:
        if data.resize is not None:
            return data.resize
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise ValueError("Cannot infer source frame size for ArUco playback.")
        return width, height

    def _draw_markers(
        self,
        *,
        image,
        observations: list[ArucoMarkerObservation],
        displacement_by_marker: dict[int, tuple[float, float, float]],
    ):
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
