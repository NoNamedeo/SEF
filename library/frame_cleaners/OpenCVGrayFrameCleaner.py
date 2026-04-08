import cv2
from typing import Any, Dict

from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame

class OpenCVGrayFrameCleaner(IFrameCleaner):

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)

    def clean(self, frame: Frame) -> Frame:
        return cv2.cvtColor(frame.frame, cv2.COLOR_RGB2GRAY)

