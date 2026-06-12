from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from sef.builtin.visualizers.Matplotlib.MatplotlibArtifactVisualizer import MatplotlibArtifactVisualizer
from sef.core.artifacts.data.ArucoDisplacementData import ArucoMarkerDisplacementData
from sef.core.artifacts.data.ArucoRelativeMotionData import ArucoMarkerRelativeMotionData
from sef.core.interfaces.IData import IData
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.interfaces.StreamingContracts import IStreamingVisualizer
from sef.core.visualization.VisualArtifact import VisualArtifact
from sef.core.visualization.VisualizationContext import VisualizationContext


class MatplotlibArucoMotionVisualizer(MatplotlibArtifactVisualizer, IStreamingVisualizer):
    """Render displacement or relative-motion timelines for ArUco markers."""

    capabilities = StageCapabilities.streaming(
        stateful=True,
        preserves_order=True,
        realtime_safe=False,
    )

    def __init__(self, config=None):
        super().__init__(config)
        self.figure_size = self.config.get("figure_size", (12, 9))
        self.grid = bool(self.config.get("grid", True))
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")
        self.grid_color = self.config.get("grid_color", "#30363d")
        self.palette = tuple(
            self.config.get(
                "palette",
                ["#58a6ff", "#7ee787", "#ffa657", "#d2a8ff", "#ff7b72", "#79c0ff"],
            )
        )

    def render(
        self,
        data,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        if isinstance(data, ArucoMarkerDisplacementData):
            return (self._render_displacement(data, context),)
        if isinstance(data, ArucoMarkerRelativeMotionData):
            return (self._render_relative_motion(data, context),)
        raise TypeError(
            "MatplotlibArucoMotionVisualizer requires ArucoMarkerDisplacementData "
            f"or ArucoMarkerRelativeMotionData, got {type(data).__name__}."
        )

    def render_stream(
        self,
        data: Iterable[IData],
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        displacement_data = ArucoMarkerDisplacementData.from_stream_items(data)
        if not displacement_data.series:
            return ()
        return self.render(displacement_data, context)

    def _render_displacement(
        self,
        data: ArucoMarkerDisplacementData,
        context: VisualizationContext | None,
    ) -> VisualArtifact:
        fig = self._create_motion_figure(data.title, rows=3)
        axes = fig.axes
        metric_labels = (
            ("displacement_x", "dx [px]"),
            ("displacement_y", "dy [px]"),
            ("displacement_magnitude", "|d| [px]"),
        )
        for axis, (attribute_name, axis_label) in zip(axes, metric_labels):
            for index, series in enumerate(data.series):
                x_values = self._timeline(series.frame_indices, series.timestamps, series.metadata)
                y_values = np.asarray(getattr(series, attribute_name), dtype=float)
                axis.plot(
                    x_values,
                    y_values,
                    linewidth=2,
                    color=self.palette[index % len(self.palette)],
                    label=f"Marker {series.marker_id}",
                )
            self._style_axis(axis, axis_label)

        axes[0].set_title(data.title, color=self.text_color)
        axes[-1].set_xlabel(self._time_axis_label(data.series), color=self.text_color)
        legend = axes[0].legend(facecolor=self.axes_facecolor, edgecolor=self.grid_color)
        for text in legend.get_texts():
            text.set_color(self.text_color)

        return self._build_image_artifact(
            fig,
            title=data.title,
            description="Marker displacement over time.",
            metadata=self._artifact_metadata(context, {"data_type": type(data).__name__}),
        )

    def _render_relative_motion(
        self,
        data: ArucoMarkerRelativeMotionData,
        context: VisualizationContext | None,
    ) -> VisualArtifact:
        fig = self._create_motion_figure(data.title, rows=1)
        axis = fig.axes[0]
        for index, series in enumerate(data.series):
            x_values = self._timeline(series.frame_indices, series.timestamps, series.metadata)
            y_values = np.asarray(series.distance_deltas, dtype=float)
            axis.plot(
                x_values,
                y_values,
                linewidth=2,
                color=self.palette[index % len(self.palette)],
                label=f"Pair {series.marker_pair[0]}-{series.marker_pair[1]}",
            )

        self._style_axis(axis, "Distance delta [px]")
        axis.set_title(data.title, color=self.text_color)
        axis.set_xlabel(self._time_axis_label(data.series), color=self.text_color)
        legend = axis.legend(facecolor=self.axes_facecolor, edgecolor=self.grid_color)
        for text in legend.get_texts():
            text.set_color(self.text_color)

        return self._build_image_artifact(
            fig,
            title=data.title,
            description="Relative marker distance variation over time.",
            metadata=self._artifact_metadata(context, {"data_type": type(data).__name__}),
        )

    def _create_motion_figure(self, title: str, rows: int):
        fig, axis = self._create_figure(
            figure_size=self.figure_size,
            figure_facecolor=self.figure_facecolor,
        )
        fig.clear()
        axes = fig.subplots(rows, 1, sharex=True)
        if rows == 1:
            axes = [axes]
        for axis in axes:
            axis.set_facecolor(self.axes_facecolor)
        fig.suptitle(title, color=self.text_color)
        fig.tight_layout()
        return fig

    def _style_axis(self, axis, y_label: str) -> None:
        axis.tick_params(colors=self.text_color)
        axis.xaxis.label.set_color(self.text_color)
        axis.yaxis.label.set_color(self.text_color)
        axis.title.set_color(self.text_color)
        axis.set_ylabel(y_label)
        if self.grid:
            axis.grid(True, color=self.grid_color, linestyle="--", alpha=0.5)

    @staticmethod
    def _timeline(
        frame_indices: list[int],
        timestamps: list[float | None],
        metadata: dict[str, object] | None = None,
    ) -> np.ndarray:
        use_timestamps = bool((metadata or {}).get("use_timestamps", True))
        if use_timestamps and timestamps and all(timestamp is not None and math.isfinite(timestamp) for timestamp in timestamps):
            return np.asarray(timestamps, dtype=float)
        return np.asarray(frame_indices, dtype=float)

    @staticmethod
    def _time_axis_label(series_collection) -> str:
        for series in series_collection:
            timestamps = getattr(series, "timestamps", [])
            use_timestamps = bool(getattr(series, "metadata", {}).get("use_timestamps", True))
            if use_timestamps and timestamps and all(timestamp is not None and math.isfinite(timestamp) for timestamp in timestamps):
                return "Time [s]"
        return "Frame Index"
