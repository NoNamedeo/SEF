import numpy as np

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.ISignal import ISignal
from library.core.artifacts.Data import Data
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer


class VerticalPositionAnalyzer(IAnalyzer):

    def __init__(self, config=None):
        super().__init__(config)

    def analyze(self, signal: ISignal):
        if not signal.signal:
            return np.array([]), np.array([])

        max_frame = max(item['frame_number'] for item in signal.signal)

        y_positions = np.full((max_frame + 1,), np.nan, dtype=float)
        frames = np.arange(max_frame + 1, dtype=int)

        for item in signal.signal:
            frame_number = item['frame_number']
            centroid = item['centroid']
            if centroid is not None:
                y_positions[frame_number] = centroid[1]

        MatplotlibFunctionVisualizer.visualize(frames, y_positions)

        return Data({
            'y_positions': y_positions,
            'frames': frames,
        })

