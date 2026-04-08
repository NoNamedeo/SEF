import cv2
from typing import Any, Dict
from library.core.abstractions.IFrameExtractor import IFrameExtractor
from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer

class OpenCVBufferedFrameExtractor(IFrameExtractor):
    def __init__(self, buffer: FrameBuffer, path: str, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.path = path
        self.buffer = buffer

    def extract(self, frame_cleaners: list[IFrameCleaner]):
        cap = cv2.VideoCapture(self.path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.path}")

        try:
            for frame in self._frames(cap):
                for cleaner in frame_cleaners:
                    frame = cleaner.clean(frame)
                self.buffer.put(frame)
        finally:
            self.buffer.close()
            cap.release()

    @staticmethod
    def _frames(cap):
        ret, frame = cap.read()
        while ret:
            yield Frame(frame)
            ret, frame = cap.read()
