from __future__ import annotations

import numpy as np

from sef.builtin.visualizers.Matplotlib.MatplotlibArtifactVisualizer import MatplotlibArtifactVisualizer
from sef.core.artifacts.data.CategoryData import CategoryData
from sef.core.visualization.VisualArtifact import VisualArtifact
from sef.core.visualization.VisualizationContext import VisualizationContext


class MatplotlibHistogramVisualizer(MatplotlibArtifactVisualizer):
    """Render category counts into a PNG bar chart artifact."""

    def __init__(self, config=None):
        super().__init__(config)
        self.figure_size = self.config.get("figure_size", (10, 6))
        self.grid = bool(self.config.get("grid", True))
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")
        self.grid_color = self.config.get("grid_color", "#30363d")
        self.bar_color = self.config.get("bar_color", "#58a6ff")

    def render(
        self,
        data: CategoryData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        categories = list(data.category_counts.keys())
        values = [data.category_counts[category] for category in categories]
        x = np.arange(len(categories))

        fig, ax = self._create_figure(
            figure_size=self.figure_size,
            figure_facecolor=self.figure_facecolor,
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

        for index, value in enumerate(values):
            ax.text(index, value, str(value), ha="center", va="bottom", color=self.text_color, fontsize=9)

        fig.tight_layout()
        artifact = self._build_image_artifact(
            fig,
            title="Category Distribution",
            metadata=self._artifact_metadata(context, {"data_type": type(data).__name__}),
        )
        return (artifact,)
