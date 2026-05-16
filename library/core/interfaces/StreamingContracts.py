from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable

from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from library.core.interfaces.IFrameExporter import FrameExportContext, IFrameExporter
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.ISignalSample import ISignalSample
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.pipeline.LatencyPolicy import FrameLatencyPolicy
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from library.core.visualization.VisualArtifact import VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext


class IStreamingFrameExtractor(IFrameExtractor):
    """Frame extractor that can publish frames into a bounded output stream."""

    @abstractmethod
    def extract_into(
        self,
        output_buffer: FrameBuffer,
        latency_policy: FrameLatencyPolicy,
    ) -> None:
        """Read frames and publish accepted frames into ``output_buffer``."""


class IStreamingFrameBufferProcessor(IFrameBufferProcessor):
    """Frame processor that can transform a stream without materializing it."""

    @abstractmethod
    def process_into(
        self,
        input_buffer: FrameBuffer,
        output_buffer: FrameBuffer,
        *,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore | None,
    ) -> None:
        """Consume ``input_buffer`` and publish processed frames to ``output_buffer``."""


class IStreamingFrameExporter(IFrameExporter):
    """Frame exporter that writes artifacts while forwarding the frame stream."""

    @abstractmethod
    def export_into(
        self,
        frames: FrameBuffer,
        output_buffer: FrameBuffer,
        context: FrameExportContext,
    ) -> tuple[VisualArtifact, ...]:
        """Export frames and forward each original frame to ``output_buffer``."""


class IStreamingSignalExtractor(ISignalExtractor):
    """Signal extractor that emits samples progressively from frame input."""

    @abstractmethod
    def extract_into(self, frames: FrameBuffer, output_buffer: SignalBuffer) -> None:
        """Consume frames and publish signal samples to ``output_buffer``."""


class IStreamingSignalCleaner(ISignalCleaner):
    """Signal cleaner that can transform samples progressively."""

    @abstractmethod
    def clean_into(self, input_signal: Iterable[ISignalSample], output_buffer: SignalBuffer) -> None:
        """Consume signal samples and publish cleaned samples to ``output_buffer``."""


class IStreamingAnalyzer(IAnalyzer):
    """Analyzer that can publish progressive data while still returning a final result."""

    @abstractmethod
    def analyze_into(self, signal: Iterable[ISignalSample], output_buffer: DataBuffer) -> IData:
        """Consume signal samples, publish progressive data, and return final data."""


class IStreamingVisualizer(IVisualizer):
    """Visualizer that can consume progressive analyzer data."""

    @abstractmethod
    def render_stream(
        self,
        data: Iterable[IData],
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        """Render from a data stream subscription."""
