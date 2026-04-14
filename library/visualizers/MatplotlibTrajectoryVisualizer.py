from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from library.core.artifacts.TrajectoryData import TrajectoryData
from library.core.interfaces.IVisualizer import IVisualizer


class MatplotlibTrajectoryVisualizer(IVisualizer):
    """
    Visualizes multi-point trajectories from optical flow tracking.
    """

    def __init__(self, config=None):
        super().__init__(config)

        self.figure_size = self.config.get("figure_size", (10, 6))
        self.grid = bool(self.config.get("grid", True))
        self.show = bool(self.config.get("show", False))

        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")

        self.cmap = self.config.get("cmap", "viridis")

        self.show_points = bool(self.config.get("show_points", True))
        self.line_width = float(self.config.get("line_width", 1.5))
        self.point_size = float(self.config.get("point_size", 25))

    def visualize(self, data: TrajectoryData):
        fig, ax = plt.subplots(
            figsize=self.figure_size,
            facecolor=self.figure_facecolor,
        )
        ax.set_facecolor(self.axes_facecolor)

        trajectories_x = data.trajectories_x
        trajectories_y = data.trajectories_y

        num_tracks = len(trajectories_x)

        colors = plt.cm.get_cmap(self.cmap, num_tracks)

        for i in range(num_tracks):
            x = np.asarray(trajectories_x[i], dtype=float)
            y = np.asarray(trajectories_y[i], dtype=float)

            if len(x) == 0:
                continue

            color = colors(i)

            # trajectory line
            ax.plot(
                x,
                y,
                color=color,
                linewidth=self.line_width,
                alpha=0.9,
                label=f"Track {i}",
            )

            # current point
            if self.show_points:
                ax.scatter(
                    x[-1],
                    y[-1],
                    color=color,
                    s=self.point_size,
                    edgecolors="white",
                    linewidths=0.5,
                )

            # optional fading effect (old points lighter)
            ax.scatter(
                x,
                y,
                color=color,
                s=10,
                alpha=0.15,
            )

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

        if self.show:
            plt.show()

        return fig, ax
