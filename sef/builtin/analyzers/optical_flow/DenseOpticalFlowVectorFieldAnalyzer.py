from __future__ import annotations

from library.core.artifacts.signal_sample.DenseOpticalFlowSignalSample import (
    DenseOpticalFlowSignalSample,
)
from library.core.artifacts.data.VectorFieldGraphData import VectorFieldGraphData
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.ISignal import ISignal


class DenseOpticalFlowVectorFieldAnalyzer(IAnalyzer):
    """
    Converts dense optical flow samples into vector field data.
    """

    def __init__(self, config=None):
        super().__init__(config)

        # quale frame usare (default: ultimo)
        self.sample_index = self.config.get("sample_index", -1)

        # normalizza vettori (utile per visualizzazione)
        self.normalize = bool(self.config.get("normalize", False))

    def analyze(self, signal: ISignal) -> IData:
        if len(signal) == 0:
            raise ValueError("Signal is empty")

        sample = signal.samples[self.sample_index]

        if not isinstance(sample, DenseOpticalFlowSignalSample):
            raise TypeError("Expected DenseOpticalFlowSignalSample")

        rows, cols = sample.grid_shape

        if rows == 0 or cols == 0:
            raise ValueError("Sample contains no grid data")

        x_coords: list[float] = []
        y_coords: list[float] = []
        u: list[float] = []
        v: list[float] = []

        cell_size = sample.cell_size

        for idx, (dx, dy) in enumerate(sample.motion_field):
            r = idx // cols
            c = idx % cols

            # centro della cella
            x = c * cell_size + cell_size / 2
            y = r * cell_size + cell_size / 2

            if self.normalize:
                norm = (dx**2 + dy**2) ** 0.5
                if norm > 0:
                    dx /= norm
                    dy /= norm

            x_coords.append(float(x))
            y_coords.append(float(y))
            u.append(float(dx))
            v.append(float(dy))

        return VectorFieldGraphData(
            x=x_coords,
            y=y_coords,
            u=u,
            v=v,
            title="Dense Optical Flow Vector Field",
            x_label="X [px]",
            y_label="Y [px]",
            metadata={
                "rows": rows,
                "cols": cols,
                "cell_size": cell_size,
                "normalized": self.normalize,
            },
        )
