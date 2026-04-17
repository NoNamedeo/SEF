from __future__ import annotations

import numpy as np
from matplotlib import cm

from library.core.artifacts.TrajectoryData import TrajectoryData
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext
from library.visualizers.MatplotlibArtifactVisualizer import MatplotlibArtifactVisualizer


class MatplotlibTrajectoryVisualizer(MatplotlibArtifactVisualizer):
    """Visualize multi-point trajectories as a PNG artifact."""

    def __init__(self, config=None):
        super().__init__(config)
        self.figure_size = self.config.get("figure_size", (10, 6))
        self.grid = bool(self.config.get("grid", True))
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")
        self.cmap = self.config.get("cmap", "viridis")
        self.show_points = bool(self.config.get("show_points", True))
        self.line_width = float(self.config.get("line_width", 1.5))
        self.point_size = float(self.config.get("point_size", 25))

    def render(
        self,
        data: TrajectoryData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        fig, ax = self._create_figure(
            figure_size=self.figure_size,
            figure_facecolor=self.figure_facecolor,
        )
        ax.set_facecolor(self.axes_facecolor)

        trajectories_x = data.trajectories_x
        trajectories_y = data.trajectories_y
        num_tracks = len(trajectories_x)
        colors = cm.get_cmap(self.cmap, max(1, num_tracks))

        for index in range(num_tracks):
            x = np.asarray(trajectories_x[index], dtype=float)
            y = np.asarray(trajectories_y[index], dtype=float)
            if len(x) == 0:
                continue

            color = colors(index)
            ax.plot(x, y, color=color, linewidth=self.line_width, alpha=0.9, label=f"Track {index}")

            if self.show_points:
                ax.scatter(
                    x[-1],
                    y[-1],
                    color=color,
                    s=self.point_size,
                    edgecolors="white",
                    linewidths=0.5,
                )

            ax.scatter(x, y, color=color, s=10, alpha=0.15)

        if self.grid:
            ax.grid(True, linestyle="--", alpha=0.3)

        ax.set_title("Optical Flow Trajectories", color=self.text_color)
        ax.set_xlabel("X Position", color=self.text_color)
        ax.set_ylabel("Y Position", color=self.text_color)
        ax.tick_params(colors=self.text_color)

        legend = ax.legend(facecolor=self.axes_facecolor, edgecolor="none")
        if legend:
            for text in legend.get_texts():
                text.set_color(self.text_color)

        fig.tight_layout()
        artifact = self._build_image_artifact(
            fig,
            title="Trajectory",
            metadata=self._artifact_metadata(context, {"data_type": type(data).__name__}),
        )
        return (artifact,)
