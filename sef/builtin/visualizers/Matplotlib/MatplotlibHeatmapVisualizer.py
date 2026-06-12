from __future__ import annotations

import numpy as np

from library.core.artifacts.data.VectorFieldGraphData import VectorFieldGraphData
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext
from library.visualizers.Matplotlib.MatplotlibArtifactVisualizer import MatplotlibArtifactVisualizer


class MatplotlibHeatmapVisualizer(MatplotlibArtifactVisualizer):
    """Render vector field derivatives as a PNG heatmap artifact."""

    def __init__(self, config=None):
        super().__init__(config)
        self.figure_size = self.config.get("figure_size", (10, 6))
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")
        self.mode = self.config.get("mode", "magnitude")
        self.cmap = self.config.get("cmap", "viridis")
        self.show_colorbar = bool(self.config.get("colorbar", True))

    def render(
        self,
        data: VectorFieldGraphData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        u = np.asarray(data.u, dtype=float)
        v = np.asarray(data.v, dtype=float)
        rows = data.metadata.get("rows")
        cols = data.metadata.get("cols")

        if rows is None or cols is None:
            raise ValueError("Missing 'rows' and 'cols' in metadata.")
        if len(u) != rows * cols:
            raise ValueError("Inconsistent grid size for heatmap rendering.")

        u_grid = u.reshape(rows, cols)
        v_grid = v.reshape(rows, cols)
        if self.mode == "magnitude":
            z = np.sqrt(u_grid**2 + v_grid**2)
            suffix = "Flow Magnitude"
        elif self.mode == "x":
            z = u_grid
            suffix = "Flow X Component"
        elif self.mode == "y":
            z = v_grid
            suffix = "Flow Y Component"
        elif self.mode == "angle":
            z = np.arctan2(v_grid, u_grid)
            suffix = "Flow Angle"
        else:
            raise ValueError(f"Unknown heatmap mode: {self.mode}")

        fig, ax = self._create_figure(
            figure_size=self.figure_size,
            figure_facecolor=self.figure_facecolor,
        )
        ax.set_facecolor(self.axes_facecolor)
        image = ax.imshow(z, cmap=self.cmap, origin="upper", aspect="auto")

        if self.show_colorbar:
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.ax.yaxis.set_tick_params(color=self.text_color)
            for label in colorbar.ax.get_yticklabels():
                label.set_color(self.text_color)

        ax.tick_params(colors=self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.title.set_color(self.text_color)
        ax.set_title(f"{data.title} ({suffix})")
        ax.set_xlabel(data.x_label)
        ax.set_ylabel(data.y_label)

        fig.tight_layout()
        artifact = self._build_image_artifact(
            fig,
            title=data.title or "Heatmap",
            metadata=self._artifact_metadata(
                context,
                {"data_type": type(data).__name__, "mode": self.mode},
            ),
        )
        return (artifact,)
