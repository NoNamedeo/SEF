from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from library.core.abstractions.IVisualizer import IVisualizer
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData


class MatplotlibFunctionVisualizer(IVisualizer):
    """Render analyzer output with Matplotlib."""

    def __init__(self, config=None):
        super().__init__(config)
        self.figure_size = self.config.get("figure_size", (10, 6))
        self.show_scatter = bool(self.config.get("show_scatter", True))
        self.grid = bool(self.config.get("grid", True))
        self.show = bool(self.config.get("show", False))
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")
        self.grid_color = self.config.get("grid_color", "#30363d")
        self.scatter_color = self.config.get("scatter_color", "#7ee787")
        self.line_color = self.config.get("line_color", "#58a6ff")

    def visualize(self, data: TwoDimGraphData):
        x = np.asarray(data.x, dtype=float)
        y = np.asarray(data.y, dtype=float)

        fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.figure_facecolor)
        ax.set_facecolor(self.axes_facecolor)
        ax.plot(x, y, color=self.line_color, linewidth=2, label=data.label)

        if self.show_scatter:
            ax.scatter(x, y, color=self.scatter_color, s=30, alpha=0.7, label="Samples")

        if self.grid:
            ax.grid(True, color=self.grid_color, linestyle="--", alpha=0.5)

        ax.tick_params(colors=self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.title.set_color(self.text_color)
        ax.set_xlabel(data.x_label)
        ax.set_ylabel(data.y_label)
        ax.set_title(data.title)

        legend = ax.legend(facecolor=self.axes_facecolor, edgecolor=self.grid_color)
        for text in legend.get_texts():
            text.set_color(self.text_color)

        fig.tight_layout()
        if self.show:
            plt.show()
        return fig, ax
