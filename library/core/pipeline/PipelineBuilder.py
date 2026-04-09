from typing import Any

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.artifacts.CompositeFrameCleaner import CompositeFrameCleaner
from library.core.abstractions.IFrameExtractor import IFrameExtractor
from library.core.abstractions.ISignalCleaner import ISignalCleaner
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.pipeline.Pipeline import Pipeline
from library.core.validators.pipeline.IPipelineValidator import IPipelineValidator


class PipelineBuilder:
    def __init__(self):
        self.pipeline = Pipeline()

    def with_frame_extractor(self, extractor: IFrameExtractor):
        """
        Sets the frame extractor to be used in the pipeline.

        :param extractor: The frame extractor to use.
        :type extractor: IFrameExtractor
        :return: self
        :rtype: PipelineBuilder
        """
        self.pipeline.frame_extractor = extractor
        return self

    def with_composite_frame_cleaner(self, composite_frame_cleaner: CompositeFrameCleaner):
        """
        Sets the composite frame cleaner to be used in the pipeline.

        :param composite_frame_cleaner: The composite frame cleaner to use.
        :type composite_frame_cleaner: CompositeFrameCleaner
        :return: self
        :rtype: PipelineBuilder
        """
        self.pipeline.frame_cleaner = composite_frame_cleaner
        return self

    def with_signal_extractor(self, extractor: ISignalExtractor):
        """
        Sets the signal extractor to be used in the pipeline.

        :param extractor: The signal extractor to use.
        :type extractor: ISignalExtractor
        :return: self
        :rtype: PipelineBuilder
        """
        self.pipeline.signal_extractor = extractor
        return self

    def with_signal_cleaner(self, cleaners: list[ISignalCleaner]):
        """
        Sets the signal cleaners to be used in the pipeline.

        :param cleaners: A list of signal cleaners to use.
        :type cleaners: list[ISignalCleaner]
        :return: self
        :rtype: PipelineBuilder
        """
        self.pipeline.signal_cleaners = cleaners
        return self

    def with_analyzer(self, analyzers: list[IAnalyzer]):
        """
        Sets the analyzers to be used in the pipeline.

        :param analyzers: A list of analyzers to use.
        :type analyzers: list[IAnalyzer]
        :return: self
        :rtype: PipelineBuilder
        """
        self.pipeline.analyzers = analyzers
        return self
    
    def with_validator(self, validator: IPipelineValidator):
        """
        Sets the validator to be used in the pipeline.

        :param validator: The validator to use.
        :type validator: IPipelineValidator
        :return: self
        :rtype: PipelineBuilder
        """
        self.pipeline._validator = validator
        return self

    def build(self) -> Pipeline:
        """
        Builds and returns the pipeline.

        :return: The pipeline
        :rtype: Pipeline
        """
        return self.pipeline