from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.IFrameExtractor import IFrameExtractor
from library.core.abstractions.ISignalCleaner import ISignalCleaner
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.artifacts.CompositeFrameCleaner import CompositeFrameCleaner
from library.core.utils.NoOpFrameCleaner import NoOpFrameCleaner


class Pipeline:
    """
    Pipeline:
    - frame_extractor: must be not None
    - composite_frame_cleaner: optional
    - signal_extractor: must be not None
    - signal_cleaners: optional
    - analyzers: must be not an empty list
    """

    def __init__(self):
        self.frame_extractor: IFrameExtractor | None = None
        self.composite_frame_cleaner: CompositeFrameCleaner | None = None
        self.signal_extractor: ISignalExtractor | None = None
        self.signal_cleaners: list[ISignalCleaner] = []
        self.analyzers: list[IAnalyzer] = []

    def run(self):
        self._validate_pipeline()

        cleaner = self.composite_frame_cleaner or NoOpFrameCleaner()
        buffer = self.frame_extractor.extract(cleaner)

        signal = self.signal_extractor.extract(buffer)

        for signal_cleaner in self.signal_cleaners:
            signal = signal_cleaner.clean(signal)

        data_list = [analyzer.analyze(signal) for analyzer in self.analyzers]

        return data_list

    def _validate_pipeline(self):
        if not self.frame_extractor:
            raise ValueError("Frame extractor not valid, this step must be initialized")
        if not self.signal_extractor:
            raise ValueError("Signal extractor not valid, this step must be initialized")
        if not self.analyzers:
            raise ValueError("Analyzer list not valid, this step must be initialized")
