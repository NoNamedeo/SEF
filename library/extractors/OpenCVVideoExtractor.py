import cv2
import numpy as np
from typing import Any, Dict, Generator
from library.core.abstractions.IExtractor import IExtractor

class OpenCVVideoExtractor(IExtractor):
    """
    Estrae frame da un video usando OpenCV.

    Output:
        Generator[np.ndarray]  # frame BGR
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)

        self.resize = self.config.get("resize", None)   # (width, height)
        self.gray = self.config.get("gray", False)
        self.max_frames = self.config.get("max_frames", None)
        self.stride = self.config.get("stride", 1)      # ogni N frame

    def extract(self, path: str) -> Generator[np.ndarray, None, None]:
        """
        Parameters
        ----------
        video : str
            Path del file video

        Yields
        ------
        frame : np.ndarray
            Frame in formato OpenCV (BGR o grayscale)
        """

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        frame_count = 0
        yielded = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # stride (skip frame)
                if frame_count % self.stride != 0:
                    continue

                # resize
                if self.resize is not None:
                    frame = cv2.resize(frame, self.resize)

                # grayscale
                if self.gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                yield frame
                yielded += 1

                if self.max_frames is not None and yielded >= self.max_frames:
                    break

        finally:
            cap.release()