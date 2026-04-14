from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame


class NoOpFrameCleaner(IFrameCleaner):
    def clean(self, frame: Frame) -> Frame:
        return frame
