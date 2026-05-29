from __future__ import annotations

import numpy as np

from library.core.artifacts.buffer.DataBuffer import DataSubscription
from library.core.artifacts.data.TwoDimGraphData import TwoDimGraphData
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext
from library.visualizers.Matplotlib.MatplotlibArtifactVisualizer import MatplotlibArtifactVisualizer


class MatplotlibFunctionStreamVisualizer(MatplotlibArtifactVisualizer):
    """Render 2D series data into a PNG artifact."""

    def __init__(self, config=None):
        super().__init__(config)
        self.figure_size = self.config.get("figure_size", (10, 6))
        self.show_scatter = bool(self.config.get("show_scatter", True))
        self.grid = bool(self.config.get("grid", True))
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")
        self.grid_color = self.config.get("grid_color", "#30363d")
        self.scatter_color = self.config.get("scatter_color", "#7ee787")
        self.line_color = self.config.get("line_color", "#58a6ff")

    def render(
        self,
        data: DataSubscription,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:

        graph_data = self._resolve_graph_data(data)

        x = np.asarray(graph_data.x, dtype=float)
        y = np.asarray(graph_data.y, dtype=float)

        fig, ax = self._create_figure(
            figure_size=self.figure_size,
            figure_facecolor=self.figure_facecolor,
        )

        self._apply_style(ax, x, y, graph_data)
        fig.tight_layout()

        artifact = self._build_image_artifact(
            fig,
            title=graph_data.title or graph_data.label or "Function plot",
            metadata=self._artifact_metadata(
                context,
                {"data_type": type(graph_data).__name__},
            ),
        )

        return (artifact,)

    @staticmethod
    def _resolve_graph_data(subscription) -> TwoDimGraphData:
        point_items = list(subscription)

        if not point_items:
            raise ValueError("Empty DataBuffer")

        x_values: list[float] = []
        y_values: list[float] = []
        metadata: dict = {"points": len(point_items)}

        for item in point_items:
            x_values.extend(MatplotlibFunctionStreamVisualizer._as_float_list(item.x))
            y_values.extend(MatplotlibFunctionStreamVisualizer._as_float_list(item.y))
            metadata.update(item.metadata)

        first_item = point_items[0]

        return TwoDimGraphData(
            x=x_values,
            y=y_values,
            label=first_item.label,
            title=first_item.title,
            x_label=first_item.x_label,
            y_label=first_item.y_label,
            metadata=metadata,
        )

    @staticmethod
    def _as_float_list(value) -> list[float]:
        if isinstance(value, (list, tuple, np.ndarray)):
            return [float(item) for item in value]
        return [float(value)]

    def _apply_style(self, ax, x, y, data: TwoDimGraphData) -> None:
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