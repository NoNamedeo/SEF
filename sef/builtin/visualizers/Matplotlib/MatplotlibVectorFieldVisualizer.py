from __future__ import annotations

import numpy as np

from sef.builtin.visualizers.Matplotlib.MatplotlibArtifactVisualizer import MatplotlibArtifactVisualizer
from sef.core.artifacts.data.VectorFieldGraphData import VectorFieldGraphData
from sef.core.visualization.VisualArtifact import VisualArtifact
from sef.core.visualization.VisualizationContext import VisualizationContext


class MatplotlibVectorFieldVisualizer(MatplotlibArtifactVisualizer):
    """Render vector field (quiver plot) as a PNG artifact."""

    def __init__(self, config=None):
        super().__init__(config)
        self.figure_size = self.config.get("figure_size", (10, 6))
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")
        self.grid_color = self.config.get("grid_color", "#30363d")
        self.color = self.config.get("color", "#58a6ff")
        self.scale = self.config.get("scale", None)
        self.width = self.config.get("width", 0.003)
        self.show_grid = bool(self.config.get("grid", True))
        self.color_by_magnitude = bool(self.config.get("color_by_magnitude", False))
        self.cmap = self.config.get("cmap", "viridis")

    def render(
        self,
        data: VectorFieldGraphData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        x = np.asarray(data.x, dtype=float)
        y = np.asarray(data.y, dtype=float)
        u = np.asarray(data.u, dtype=float)
        v = np.asarray(data.v, dtype=float)

        fig, ax = self._create_figure(
            figure_size=self.figure_size,
            figure_facecolor=self.figure_facecolor,
        )
        ax.set_facecolor(self.axes_facecolor)

        if self.color_by_magnitude:
            magnitude = np.sqrt(u**2 + v**2)
            quiver = ax.quiver(
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
            colorbar = fig.colorbar(quiver, ax=ax)
            colorbar.ax.yaxis.set_tick_params(color=self.text_color)
            for label in colorbar.ax.get_yticklabels():
                label.set_color(self.text_color)
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

        ax.tick_params(colors=self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.title.set_color(self.text_color)
        ax.set_xlabel(data.x_label)
        ax.set_ylabel(data.y_label)
        ax.set_title(data.title)
        ax.set_aspect("equal")

        fig.tight_layout()
        artifact = self._build_image_artifact(
            fig,
            title=data.title or "Vector field",
            metadata=self._artifact_metadata(context, {"data_type": type(data).__name__}),
        )
        return (artifact,)
