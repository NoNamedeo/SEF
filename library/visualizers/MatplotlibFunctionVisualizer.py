import numpy as np
import matplotlib.pyplot as plt

from library.core.abstractions.IData import IData

class MatplotlibFunctionVisualizer:

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

    def visualize(self, function: IData):
        x = np.array(function.x)
        y = np.array(function.y)

        # Creazione figura
        fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.figure_facecolor)
        ax.set_facecolor(self.axes_facecolor)

        # Line plot
        ax.plot(x, y, color=self.line_color, label="Function")

        # Scatter plot opzionale
        if self.show_scatter:
            ax.scatter(x, y, color=self.scatter_color, s=30, alpha=0.7, label="Points")

        # Griglia
        if self.grid:
            ax.grid(True, color=self.grid_color)

        # Label, colore testo, legenda
        ax.tick_params(colors=self.text_color)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.title.set_color(self.text_color)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Function Visualization")
        ax.legend(facecolor=self.axes_facecolor, edgecolor=self.text_color, labelcolor=self.text_color)

        # Mostra figura
        plt.show()
