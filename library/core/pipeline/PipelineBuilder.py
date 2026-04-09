from __future__ import annotations

from collections.abc import Iterable

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.ISignalCleaner import ISignalCleaner
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.artifacts.CompositeFrameCleaner import CompositeFrameCleaner
from library.core.pipeline.Pipeline import Pipeline
from library.core.validators.pipeline.IPipelineValidator import IPipelineValidator


class PipelineBuilder:
    """Fluent builder aligned with the actual Pipeline fields."""

    def __init__(self):
        self.pipeline = Pipeline()

    def with_frame_extractor(self, extractor):
        self.pipeline.frame_extractor = extractor
        return self

    def with_composite_frame_cleaner(self, composite_frame_cleaner: CompositeFrameCleaner):
        self.pipeline.composite_frame_cleaner = composite_frame_cleaner
        return self

    def with_signal_extractor(self, extractor: ISignalExtractor):
        self.pipeline.signal_extractor = extractor
        return self

    def with_signal_cleaners(self, cleaners: Iterable[ISignalCleaner]):
        self.pipeline.signal_cleaners = list(cleaners)
        return self

    def add_signal_cleaner(self, cleaner: ISignalCleaner):
        self.pipeline.signal_cleaners.append(cleaner)
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
