from __future__ import annotations

import logging
import os
import threading

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from library.core.interfaces.IVisualizer import IVisualizer
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData

log = logging.getLogger(__name__)


class MatplotlibFunctionVisualizer(IVisualizer):
    """
    Render analyzer output with Matplotlib.

    Thread-safety
    -------------
    Matplotlib's interactive backends (MacOS, TkAgg, Qt5Agg …) can only
    create GUI windows from the **main thread**.  Calling ``plt.subplots()``
    from a worker thread raises::

        RuntimeError: Cannot create a GUI FigureManager outside the main thread
        using the MacOS backend

    This class detects the calling thread at runtime:

    * **Main thread** → uses ``pyplot.subplots()`` + ``plt.show()`` as before
      (fully interactive, backward compatible).
    * **Worker thread** → creates a ``matplotlib.figure.Figure`` and attaches
      a non-interactive ``FigureCanvasAgg`` directly, bypassing the GUI
      backend entirely.  If the figure should be "shown", it is saved to a
      file (``output_dir/<title>.png``) and an INFO message is logged.

    Configuration keys (in addition to the standard ones)
    ------------------------------------------------------
    output_dir : str, default ``"output"``
        Directory for figures saved from worker threads.
        The directory is created automatically if it does not exist.
        Has no effect when running on the main thread (figure is shown
        interactively via ``plt.show()`` there).
    """

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
        self.output_dir = self.config.get("output_dir", "output")

    # ── Public API ───────────────────────────────────────────────────────────

    def visualize(self, data: TwoDimGraphData):
        """
        Render *data* and, if ``show=True``, display or save the figure.

        Returns
        -------
        tuple[Figure, Axes]
        """
        if threading.current_thread() is threading.main_thread():
            return self._visualize_interactive(data)
        else:
            return self._visualize_offscreen(data)

    # ── Interactive path (main thread only) ──────────────────────────────────

    def _visualize_interactive(self, data: TwoDimGraphData):
        """Use pyplot — only safe from the main thread."""
        import matplotlib.pyplot as plt  # lazy import: GUI backend loaded on demand

        x = np.asarray(data.x, dtype=float)
        y = np.asarray(data.y, dtype=float)

        fig, ax = plt.subplots(figsize=self.figure_size, facecolor=self.figure_facecolor)
        self._apply_style(ax, x, y, data)
        fig.tight_layout()

        if self.show:
            plt.show()

        return fig, ax

    # ── Offscreen path (worker threads) ──────────────────────────────────────

    def _visualize_offscreen(self, data: TwoDimGraphData):
        """
        Use Agg backend directly — safe from **any** thread.

        ``matplotlib.figure.Figure`` + ``FigureCanvasAgg`` never touch the
        GUI event loop, so they work correctly on secondary pipeline threads.
        """
        x = np.asarray(data.x, dtype=float)
        y = np.asarray(data.y, dtype=float)

        fig = Figure(figsize=self.figure_size, facecolor=self.figure_facecolor)
        FigureCanvasAgg(fig)  # attach non-interactive canvas (required for savefig)
        ax = fig.add_subplot(111)
        self._apply_style(ax, x, y, data)
        fig.tight_layout()

        # Persist to disk when the caller would have shown a window
        if self.show or self.output_dir:
            self._save_figure(fig, data.title or "plot")

        return fig, ax

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _apply_style(self, ax, x, y, data: TwoDimGraphData) -> None:
        """Apply common styling — works with both pyplot Axes and Figure Axes."""
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

    def _save_figure(self, fig: Figure, title: str) -> None:
        """Save *fig* to ``{output_dir}/{safe_title}.png``."""
        safe = (
            "".join(c if (c.isalnum() or c in "._- ") else "_" for c in title)
            .strip()
            .replace(" ", "_")
        )

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{safe}.png")
        fig.savefig(path, bbox_inches="tight")
        log.info(
            "MatplotlibFunctionVisualizer [worker thread]: "
            "GUI unavailable — figure saved to '%s'.",
            path,
        )
