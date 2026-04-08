from typing import Tuple

import cv2
import numpy as np

from library.core.abstractions.ISignal import ISignal
from library.core.abstractions.ISignalCleaner import ISignalCleaner
from library.core.artifacts.Signal import Signal


class OpenCVMovingAverageCleaner(ISignalCleaner):

    def __init__(self, window_size: int, config=None):
        super().__init__(config)
        self.window_size = window_size

    def clean(self, signal: ISignal) -> ISignal:

        data = signal.signal
        cleaned = []

        # estrai i centroidi
        centroids = [item['centroid'] for item in data]

        for i in range(len(data)):
            # definisci la finestra
            start = max(0, i - self.window_size // 2)
            end = min(len(data), i + self.window_size // 2 + 1)

            # centroidi validi nella finestra (evita None)
            window_points = [c for c in centroids[start:end] if c is not None]

            if window_points:
                # media sui punti x e y separati
                avg_x = sum(p[0] for p in window_points) / len(window_points)
                avg_y = sum(p[1] for p in window_points) / len(window_points)
                smoothed_centroid: Tuple[float, float] = (avg_x, avg_y)
            else:
                smoothed_centroid = None

            # copia il box originale
            cleaned.append({
                'frame_idx': data[i]['frame_idx'],
                'box': data[i]['box'],
                'centroid': smoothed_centroid
            })

        return Signal(cleaned)
