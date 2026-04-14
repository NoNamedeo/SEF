from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from library.core.artifacts.VectorFieldGraphData import VectorFieldGraphData
from library.core.interfaces.IVisualizer import IVisualizer


class MatplotlibVectorFieldVisualizer(IVisualizer):
    """Render vector field (quiver plot) with Matplotlib."""

    def __init__(self, config=None):
        super().__init__(config)

        self.figure_size = self.config.get("figure_size", (10, 6))
        self.show = bool(self.config.get("show", False))

        # stile (coerente con il tuo visualizer)
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")
        self.grid_color = self.config.get("grid_color", "#30363d")

        # quiver options
        self.color = self.config.get("color", "#58a6ff")
        self.scale = self.config.get("scale", None)  # None = auto
        self.width = self.config.get("width", 0.003)
        self.show_grid = bool(self.config.get("grid", True))

        # coloring by magnitude
        self.color_by_magnitude = bool(self.config.get("color_by_magnitude", False))
        self.cmap = self.config.get("cmap", "viridis")

    def visualize(self, data: VectorFieldGraphData):
        x = np.asarray(data.x, dtype=float)
        y = np.asarray(data.y, dtype=float)
        u = np.asarray(data.u, dtype=float)
        v = np.asarray(data.v, dtype=float)

        fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.figure_facecolor)
        ax.set_facecolor(self.axes_facecolor)

        if self.color_by_magnitude:
            magnitude = np.sqrt(u**2 + v**2)

            q = ax.quiver(
                x,
                y,
                u,
                v,
                magnitude,
                angles="xy",
                scale_units="xy",
                scale=self.scale,
                cmap=self.cmap,
                width=self.width,
            )

            cbar = fig.colorbar(q, ax=ax)
            cbar.ax.yaxis.set_tick_params(color=self.text_color)
            plt.setp(cbar.ax.get_yticklabels(), color=self.text_color)

        else:
            ax.quiver(
                x,
                y,
                u,
                v,
                color=self.color,
                angles="xy",
                scale_units="xy",
                scale=self.scale,
                width=self.width,
            )

        if self.show_grid:
            ax.grid(True, color=self.grid_color, linestyle="--", alpha=0.5)

        # stile testo
        ax.tick_params(colors=self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.title.set_color(self.text_color)

        ax.set_xlabel(data.x_label)
        ax.set_ylabel(data.y_label)
        ax.set_title(data.title)

        # importante per non distorcere i vettori
        ax.set_aspect("equal")

        fig.tight_layout()

        if self.show:
            plt.show()

        return fig, ax
