import cv2
import numpy as np

from library.src.abstractions.ICleaner import ICleaner


class OpenCVMovingAverageCleaner(ICleaner):
    """
    Applica una media mobile alle posizioni del centroid restituito dal tracker.

    Input atteso:
        [
            {
                "frame_idx": int,
                "bbox": (x, y, w, h) | None,
                "centroid": (cx, cy) | None
            },
            ...
        ]

    Output:
        stessa struttura, con l'aggiunta di:
            "raw_centroid": centroid originale
            "centroid": centroid smussato
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.window_size = int(self.config.get("window_size", 5))

        if self.window_size <= 0:
            raise ValueError("window_size deve essere > 0")

    def _fill_missing_centroids(self, data):
        filled = []
        last_valid = None

        for item in data:
            centroid = item.get("centroid")
            if centroid is not None:
                last_valid = centroid
                filled.append(centroid)
            elif last_valid is not None:
                filled.append(last_valid)
            else:
                filled.append((0.0, 0.0))

        next_valid = None
        for index in range(len(filled) - 1, -1, -1):
            if data[index].get("centroid") is not None:
                next_valid = data[index]["centroid"]
            elif next_valid is not None and filled[index] == (0.0, 0.0):
                filled[index] = next_valid

        return filled

    def clean(self, data):
        if not data:
            return []

        filled_centroids = self._fill_missing_centroids(data)
        points = np.array(filled_centroids, dtype=np.float32).reshape(-1, 1, 2)

        kernel_size = (1, self.window_size)
        smoothed = cv2.blur(
            src=points,
            ksize=kernel_size,
            borderType=cv2.BORDER_REPLICATE,
        ).reshape(-1, 2)

        cleaned = []
        for item, smooth_point in zip(data, smoothed):
            cleaned_item = dict(item)
            cleaned_item["raw_centroid"] = item.get("centroid")
            cleaned_item["centroid"] = (float(smooth_point[0]), float(smooth_point[1]))
            cleaned.append(cleaned_item)

        return cleaned
