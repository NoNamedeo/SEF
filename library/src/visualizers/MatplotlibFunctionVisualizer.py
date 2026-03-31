import numpy as np
import matplotlib.pyplot as plt


class MatplotlibFunctionVisualizer:
    """
    Visualizza il risultato prodotto dall'analyzer.

    Input atteso:
        {
            "function": callable,
            "formula": str,
            "times": np.ndarray,
            "y_values": np.ndarray
        }
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.figure_size = self.config.get("figure_size", (10, 6))
        self.sample_points = int(self.config.get("sample_points", 200))
        self.show_scatter = bool(self.config.get("show_scatter", True))
        self.grid = bool(self.config.get("grid", True))
        self.figure_facecolor = self.config.get("figure_facecolor", "#101418")
        self.axes_facecolor = self.config.get("axes_facecolor", "#161b22")
        self.text_color = self.config.get("text_color", "#e6edf3")
        self.grid_color = self.config.get("grid_color", "#30363d")
        self.scatter_color = self.config.get("scatter_color", "#7ee787")
        self.line_color = self.config.get("line_color", "#58a6ff")

    def visualize(self, analysis, title=None, scatter_label="Dati osservati", line_label=None):
        if "function" not in analysis:
            raise ValueError("analysis deve contenere una funzione in 'function'")
        if "times" not in analysis or "y_values" not in analysis:
            raise ValueError("analysis deve contenere 'times' e 'y_values'")

        times = np.asarray(analysis["times"], dtype=np.float32)
        y_values = np.asarray(analysis["y_values"], dtype=np.float32)

        if times.size == 0 or y_values.size == 0:
            raise ValueError("Nessun dato disponibile da visualizzare")

        function = analysis["function"]
        formula = analysis.get("formula", "y(t)")
        plot_title = title or "Andamento di y nel tempo"
        line_label = line_label or formula

        if times.size == 1:
            plot_times = np.array([times[0], times[0] + 1.0], dtype=np.float32)
        else:
            plot_times = np.linspace(times.min(), times.max(), self.sample_points, dtype=np.float32)

        plot_y = function(plot_times)

        fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.figure_facecolor)
        ax.set_facecolor(self.axes_facecolor)

        if self.show_scatter:
            ax.scatter(times, y_values, color=self.scatter_color, label=scatter_label)

        ax.plot(plot_times, plot_y, color=self.line_color, linewidth=2, label=line_label)
        ax.set_title(plot_title)
        ax.set_xlabel("Tempo [s]")
        ax.set_ylabel("Posizione y [px]")
        ax.title.set_color(self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.tick_params(colors=self.text_color)

        for spine in ax.spines.values():
            spine.set_color(self.grid_color)

        if self.grid:
            ax.grid(True, linestyle="--", alpha=0.5, color=self.grid_color)

        legend = ax.legend(facecolor=self.axes_facecolor, edgecolor=self.grid_color)
        for text in legend.get_texts():
            text.set_color(self.text_color)

        fig.tight_layout()
        plt.show()

        return fig, ax
