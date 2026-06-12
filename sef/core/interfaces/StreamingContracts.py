from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable

from sef.core.artifacts.Frame import Frame
from sef.core.interfaces.BufferContracts import IBuffer, IFrameBuffer
from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.IData import IData
from sef.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from sef.core.interfaces.IFrameExporter import FrameExportContext, IFrameExporter
from sef.core.interfaces.IFrameExtractor import IFrameExtractor
from sef.core.interfaces.ISignalCleaner import ISignalCleaner
from sef.core.interfaces.ISignalExtractor import ISignalExtractor
from sef.core.interfaces.ISignalSample import ISignalSample
from sef.core.interfaces.IVisualizer import IVisualizer
from sef.core.pipeline.IntermediateFrameCapture import IntermediateFrameArtifactStore
from sef.core.pipeline.LatencyPolicy import FrameLatencyPolicy
from sef.core.visualization.VisualArtifact import VisualArtifact
from sef.core.visualization.VisualizationContext import VisualizationContext


class IStreamingFrameExtractor(IFrameExtractor):
    """
    Frame extractor that can publish frames into a bounded output stream.

    Implementations must close `output_buffer` on normal completion and should
    cooperate with the supplied latency policy instead of writing directly with
    unbounded buffering.
    """

    @abstractmethod
    def extract_into(
        self,
        output_buffer: IFrameBuffer,
        latency_policy: FrameLatencyPolicy,
    ) -> None:
        """
        Read frames and publish accepted frames into `output_buffer`.

        Parameters
        ----------
        output_buffer:
            Bounded frame queue owned by the runtime.
        latency_policy:
            Per-run policy deciding whether each frame is accepted or dropped.
        """


class IStreamingFrameBufferProcessor(IFrameBufferProcessor):
    """
    Frame processor that can transform a stream without materializing it.

    Implementations should publish outputs progressively and preserve ordering
    unless their plugin documentation explicitly states otherwise.
    """

    @abstractmethod
    def process_into(
        self,
        input_buffer: Iterable[Frame],
        output_buffer: IBuffer[Frame],
        *,
        processor_index: int,
        intermediate_store: IntermediateFrameArtifactStore | None,
    ) -> None:
        """
        Consume `input_buffer` and publish processed frames.

        Implementations must close `output_buffer` when finished and should
        abort cooperatively when downstream consumers abort.
        """


class IStreamingFrameExporter(IFrameExporter):
    """
    Frame exporter that writes artifacts while forwarding the frame stream.

    This contract lets exporters produce file-backed artifacts without forcing
    the frame stream to be fully materialized before signal extraction.
    """

    @abstractmethod
    def export_into(
        self,
        frames: Iterable[Frame],
        output_buffer: IBuffer[Frame],
        context: FrameExportContext,
    ) -> tuple[VisualArtifact, ...]:
        """Export frames and forward each original frame to `output_buffer`."""


class IStreamingSignalExtractor(ISignalExtractor):
    """Signal extractor that emits samples progressively from frame input."""

    @abstractmethod
    def extract_into(self, frames: IFrameBuffer, output_buffer: IBuffer[ISignalSample]) -> None:
        """Consume frames and publish signal samples to `output_buffer`."""


class IStreamingSignalCleaner(ISignalCleaner):
    """Signal cleaner that can transform samples progressively."""

    @abstractmethod
    def clean_into(self, input_signal: Iterable[ISignalSample], output_buffer: IBuffer[ISignalSample]) -> None:
        """Consume signal samples and publish cleaned samples to `output_buffer`."""


class IStreamingAnalyzer(IAnalyzer):
    """
    Analyzer that can publish progressive data and still return a final result.

    Progressive values feed streaming visualizers during execution. The returned
    final `IData` remains the authoritative analyzer result included in
    `PipelineOutputs`.
    """

    @abstractmethod
    def analyze_into(self, signal: Iterable[ISignalSample], output_buffer: IBuffer[IData]) -> IData:
        """Consume signal samples, publish progressive data, and return final data."""


class IStreamingVisualizer(IVisualizer):
    """
    Visualizer that can consume progressive analyzer data.

    Streaming visualizers should avoid toolkit-specific blocking calls in core
    execution paths. For live UI updates, publish through adapter-safe contracts
    such as realtime sinks or return final artifacts after the stream closes.
    """

    @abstractmethod
    def render_stream(
        self,
        data: Iterable[IData],
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        """Render from a data stream subscription."""
