from library.core.abstractions.IFrameCleaner import IFrameCleaner

class NoOpFrameCleaner(IFrameCleaner):
    def clean(self, frame):
        return frame