from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.IFrameExtractor import IFrameExtractor
from library.core.abstractions.ISignalCleaner import ISignalCleaner
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.artifacts.CompositeFrameCleaner import CompositeFrameCleaner
from library.core.utils.NoOpFrameCleaner import NoOpFrameCleaner
from library.core.validators.pipeline.PipelineValidator import PipelineValidator
from library.core.validators.pipeline.IPipelineValidator import IPipelineValidator

class Pipeline:
    """
    Pipeline:
    - frame_extractor: must be not None
    - composite_frame_cleaner: optional
    - signal_extractor: must be not None
    - signal_cleaners: optional
    - analyzers: must be not an empty list
    """

    def __init__(self, validator: IPipelineValidator | None = None):
        """
        Initializes a new pipeline with the given validator.

        :param validator: Optional validator to use. If None, a default
            validator is used.
        :type validator: IPipelineValidator | None
        """
        self.frame_extractor: IFrameExtractor | None = None
        self.composite_frame_cleaner: CompositeFrameCleaner | None = None
        self.signal_extractor: ISignalExtractor | None = None
        self.signal_cleaners: list[ISignalCleaner] = []
        self.analyzers: list[IAnalyzer] = []
        self._validator = validator or PipelineValidator()

    def run(self):
        """
        Runs the pipeline.

        This method validates the pipeline using the given validator.
        Then it extracts frames from the given video using the frame extractor.
        It cleans the frames using the composite frame cleaner.
        Then it extracts signals from the frames using the signal extractor.
        It cleans the signals using the signal cleaners.
        Finally, it analyzes the signals using the analyzers.

        Returns a list of Data objects, one for each analyzer.
        """
        self._validator.validate(self)

        cleaner = self.composite_frame_cleaner or NoOpFrameCleaner()
        buffer = self.frame_extractor.extract(cleaner)

        signal = self.signal_extractor.extract(buffer)

        for signal_cleaner in self.signal_cleaners:
            signal = signal_cleaner.clean(signal)

        data_list = [analyzer.analyze(signal) for analyzer in self.analyzers]

        return data_list
