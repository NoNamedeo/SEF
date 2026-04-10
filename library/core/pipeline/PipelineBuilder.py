from __future__ import annotations

from collections.abc import Iterable

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.ISignalCleaner import ISignalCleaner
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.pipeline.Pipeline import Pipeline
from library.core.abstractions.IPipelineValidator import IPipelineValidator


class PipelineBuilder:
    """Fluent builder aligned with the actual Pipeline fields."""

    def __init__(self):
        self.pipeline = Pipeline()

    def with_frame_extractor(self, extractor):
        self.pipeline.frame_extractor = extractor
        return self

    def with_frame_cleaners(self, frame_cleaners: Iterable[IFrameCleaner]):
        self.pipeline.frame_cleaners = list(frame_cleaners)
        return self

    def add_frame_cleaner(self, frame_cleaner: IFrameCleaner):
        self.pipeline.frame_cleaners.append(frame_cleaner)
        return self

    def with_signal_extractor(self, extractor: ISignalExtractor):
        self.pipeline.signal_extractor = extractor
        return self

    def with_signal_cleaners(self, signal_cleaners: Iterable[ISignalCleaner]):
        self.pipeline.signal_cleaners = list(signal_cleaners)
        return self

    def add_signal_cleaner(self, signal_cleaner: ISignalCleaner):
        self.pipeline.signal_cleaners.append(signal_cleaner)
        return self

    def with_analyzers(self, analyzers: Iterable[IAnalyzer]):
        self.pipeline.analyzers = list(analyzers)
        return self

    def add_analyzer(self, analyzer: IAnalyzer):
        self.pipeline.analyzers.append(analyzer)
        return self

    def with_validator(self, validator: IPipelineValidator):
        self.pipeline._validator = validator
        return self

    def build(self) -> Pipeline:
        return self.pipeline
