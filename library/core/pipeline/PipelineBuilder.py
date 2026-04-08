from typing import Any

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.abstractions.IFrameExtractor import IFrameExtractor
from library.core.abstractions.ISignalCleaner import ISignalCleaner
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.abstractions.IVisualizer import IVisualizer
from library.core.pipeline.Pipeline import Pipeline


class PipelineBuilder:
    def __init__(self):
        self.pipeline = Pipeline()

    def with_frame_extractor(self, extractor: IFrameExtractor):
        self.pipeline.frame_extractor = extractor
        return self

    def with_frame_cleaner(self, cleaner: IFrameCleaner):
        self.pipeline.frame_cleaner = cleaner
        return self

    def with_signal_extractor(self, extractor: ISignalExtractor):
        self.pipeline.signal_extractor = extractor
        return self

    def with_signal_cleaner(self, cleaner: ISignalCleaner):
        self.pipeline.signal_cleaner = cleaner
        return self

    def with_analyzer(self, analyzer: IAnalyzer):
        self.pipeline.analyzer = analyzer
        return self

    def build(self) -> Pipeline:
        return self.pipeline