import cv2
import numpy as np

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.ISignal import ISignal


class OpenCVYTimeAnalyzer(IAnalyzer):

    def __init__(self, config=None):
        super().__init__(config)

    def analyze(self, signal: ISignal):
        data_list = []

        for item in signal.signal:
            frame_idx = item['frame_idx']
            centroid = item['centroid']
            if centroid is not None:
                y = centroid[1]
                data_list.append([frame_idx, y])

        data_array = np.array(data_list, dtype=float)
        return data_array

