from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.artifacts.Frame import Frame


class CompositeFrameCleaner(IFrameCleaner):

    def __init__(self, cleaners: list[IFrameCleaner]):
        super().__init__()
        self.cleaners = cleaners

    def clean(self, frame: Frame) -> Frame:
        for cleaner in self.cleaners:
            frame = cleaner.clean(frame)
        return frame
