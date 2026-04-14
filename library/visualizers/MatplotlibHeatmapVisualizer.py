from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from library.core.abstractions.IVisualizer import IVisualizer
from library.core.artifacts.VectorFieldGraphData import VectorFieldGraphData


class MatplotlibHeatmapVisualizer(IVisualizer):
    """Render vector field as a heatmap (magnitude / component / angle)."""
    #TODO: il funzionamento dipende dal fatto o no che l'analyzer metta rows e columns
    #TODO: dentro a metadata del data

    def __init__(self, config=None):
        super().__init__(config)

        self.figure_size = self.config.get("figure_size", (10, 6))
        self.show = bool(self.config.get("show", False))

        # stile
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")

        # heatmap options
        self.mode = self.config.get("mode", "magnitude")
        # "magnitude" | "x" | "y" | "angle"

        self.cmap = self.config.get("cmap", "viridis")
        self.show_colorbar = bool(self.config.get("colorbar", True))

    def visualize(self, data: VectorFieldGraphData):
        x = np.asarray(data.x, dtype=float)
        y = np.asarray(data.y, dtype=float)
        u = np.asarray(data.u, dtype=float)
        v = np.asarray(data.v, dtype=float)

        rows = data.metadata.get("rows")
        cols = data.metadata.get("cols")

        if rows is None or cols is None:
            raise ValueError("Missing 'rows' and 'cols' in metadata")

        if len(u) != rows * cols:
            raise ValueError("Inconsistent grid size")

        # costruzione della matrice
        u_grid = u.reshape(rows, cols)
        v_grid = v.reshape(rows, cols)

        if self.mode == "magnitude":
            z = np.sqrt(u_grid**2 + v_grid**2)
            title = "Flow Magnitude"
        elif self.mode == "x":
            z = u_grid
            title = "Flow X Component"
        elif self.mode == "y":
            z = v_grid
            title = "Flow Y Component"
        elif self.mode == "angle":
            z = np.arctan2(v_grid, u_grid)
            title = "Flow Angle"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.figure_facecolor)
        ax.set_facecolor(self.axes_facecolor)

        im = ax.imshow(
            z,
            cmap=self.cmap,
            origin="upper",  # coerente con coordinate immagine
            aspect="auto"
        )

        if self.show_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.yaxis.set_tick_params(color=self.text_color)
            plt.setp(cbar.ax.get_yticklabels(), color=self.text_color)

        # stile testo
        ax.tick_params(colors=self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.title.set_color(self.text_color)

        ax.set_title(f"{data.title} ({title})")
        ax.set_xlabel(data.x_label)
        ax.set_ylabel(data.y_label)

        fig.tight_layout()

        if self.show:
            plt.show()

        return fig, ax