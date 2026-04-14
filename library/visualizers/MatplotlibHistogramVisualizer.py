from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from library.core.abstractions.IVisualizer import IVisualizer
from library.core.artifacts.CategoryData import CategoryData


class MatplotlibHistogramVisualizer(IVisualizer):
    """
    Visualizes category_counts as a bar chart (histogram-like).
    """

    def __init__(self, config=None):
        super().__init__(config)

        self.figure_size = self.config.get("figure_size", (10, 6))
        self.show = bool(self.config.get("show", True))
        self.grid = bool(self.config.get("grid", True))

        # styling
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")
        self.grid_color = self.config.get("grid_color", "#30363d")
        self.bar_color = self.config.get("bar_color", "#58a6ff")

    def visualize(self, data: CategoryData):

        categories = list(data.category_counts.keys())
        values = [data.category_counts[c] for c in categories]

        x = np.arange(len(categories))

        fig, ax = plt.subplots(
            figsize=self.figure_size,
            facecolor=self.figure_facecolor
        )

        ax.set_facecolor(self.axes_facecolor)

        ax.bar(x, values, color=self.bar_color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right", color=self.text_color)

        ax.set_ylabel("Count", color=self.text_color)
        ax.set_title("Category Distribution (Barrier Crossings)", color=self.text_color)

        ax.tick_params(colors=self.text_color)

        if self.grid:
            ax.grid(True, axis="y", color=self.grid_color, linestyle="--", alpha=0.5)

        # optional: value labels on top of bars
        for i, v in enumerate(values):
            ax.text(
                i,
                v,
                str(v),
                ha="center",
                va="bottom",
                color=self.text_color,
                fontsize=9
            )

        plt.tight_layout()

        if self.show:
            plt.show()

        return fig, ax