from __future__ import annotations

from pathlib import Path

import numpy as np

from library.analyzers.NoAnalyzer import NoAnalyzer
from library.core.artifacts.BoxSignalSample import BoxSignalSample
from library.core.artifacts.DataBuffer import DataBuffer
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.SignalBuffer import SignalBuffer
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.artifacts.TwoDimPointData import TwoDimPointData
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.Pipeline import Pipeline
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.core.visualization.VisualArtifact import TextArtifact, VideoFileArtifact, VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext
from library.exporters.OpenCVFrameBufferVideoExporter import OpenCVFrameBufferVideoExporter
from library.signal_extractors.NoSignalExtractor import NoSignalExtractor


def test_frame_buffer_can_close_when_full() -> None:
    buffer = FrameBuffer(buffer_size=2)
    buffer.put(_frame(0, 0))
    buffer.put(_frame(1, 1))

    buffer.close()

    assert [frame.index for frame in buffer] == [0, 1]


def test_signal_and_data_buffers_preserve_order_for_multiple_consumers() -> None:
    signal_buffer = SignalBuffer(buffer_size=2)
    signal_buffer.set_consumer_count(2)
    first_signal_consumer = signal_buffer.subscribe(0)
    second_signal_consumer = signal_buffer.subscribe(1)

    signal_buffer.put(BoxSignalSample(frame_index=0, box=None, centroid=(0.0, 1.0)))
    signal_buffer.put(BoxSignalSample(frame_index=1, box=None, centroid=(0.0, 2.0)))
    signal_buffer.close()

    assert [sample.frame_index for sample in first_signal_consumer] == [0, 1]
    assert [sample.frame_index for sample in second_signal_consumer] == [0, 1]

    data_buffer = DataBuffer(buffer_size=2)
    data_buffer.set_consumer_count(2)
    first_data_consumer = data_buffer.subscribe(0)
    second_data_consumer = data_buffer.subscribe(1)

    data_buffer.put(TwoDimPointData(x=0.0, y=1.0))
    data_buffer.put(TwoDimPointData(x=1.0, y=2.0))
    data_buffer.close()

    assert [point.y for point in first_data_consumer] == [1.0, 2.0]
    assert [point.y for point in second_data_consumer] == [1.0, 2.0]


def test_pipeline_runs_stream_capable_components_with_live_visualizer() -> None:
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StreamingFrameExtractor(frame_count=3))
        .add_frame_processor(SingleFrameProcessorAdapter(AddOneProcessor()))
        .with_signal_extractor(StreamingSignalExtractor())
        .add_analyzer(StreamingAnalyzer())
        .add_visualizer_for_results(StreamingTextVisualizer(), [0])
        .build_context()
    )

    outputs = Pipeline(context).run()

    assert len(outputs.results) == 1
    assert isinstance(outputs.results[0], TwoDimGraphData)
    assert outputs.results[0].y == [1.0, 2.0, 3.0]
    assert len(outputs.final_artifacts) == 1
    assert isinstance(outputs.final_artifacts[0], TextArtifact)
    assert outputs.final_artifacts[0].content == "streamed=3;sum=6.0"


def test_pipeline_streams_frame_exporter_without_buffering_full_video(tmp_path: Path) -> None:
    output_path = tmp_path / "streamed.mp4"
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StreamingFrameExtractor(frame_count=4))
        .add_frame_processor(SingleFrameProcessorAdapter(AddOneProcessor()))
        .add_frame_exporter(OpenCVFrameBufferVideoExporter(output_path, fps=10.0, max_exported_frames=4))
        .with_signal_extractor(NoSignalExtractor())
        .add_analyzer(NoAnalyzer())
        .build_context()
    )

    outputs = Pipeline(context).run()

    assert len(outputs.results) == 1
    assert len(outputs.final_artifacts) == 1
    assert isinstance(outputs.final_artifacts[0], VideoFileArtifact)
    assert outputs.final_artifacts[0].metadata["frame_count"] == 4
    assert output_path.exists()
    assert output_path.stat().st_size > 0


class StreamingFrameExtractor(IFrameExtractor):
    def __init__(self, frame_count: int) -> None:
        super().__init__()
        self._frame_count = frame_count
        self.buffer = FrameBuffer(buffer_size=2)

    def extract(self) -> FrameBuffer:
        for frame_index in range(self._frame_count):
            self.buffer.put(_frame(frame_index, frame_index))
        self.buffer.close()
        return self.buffer


class AddOneProcessor(ISingleFrameProcessor):
    def process(self, frame: Frame) -> Frame:
        return Frame(
            image=frame.image + 1,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata=dict(frame.metadata),
        )


class StreamingSignalExtractor(ISignalExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = SignalBuffer(buffer_size=2)

    def extract(self, buffer: FrameBuffer) -> ISignal:
        for frame in buffer:
            value = float(frame.image[0, 0, 0])
            self.buffer.put(
                BoxSignalSample(
                    frame_index=frame.index or 0,
                    box=None,
                    centroid=(0.0, value),
                    timestamp_seconds=frame.timestamp_seconds,
                )
            )
        self.buffer.close()
        return self.buffer


class StreamingAnalyzer(IAnalyzer):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = DataBuffer(buffer_size=2)

    def analyze(self, signal: ISignal) -> IData:
        x_values: list[float] = []
        y_values: list[float] = []
        for sample in signal:
            value = float(sample.centroid[1]) if sample.centroid is not None else 0.0
            x_values.append(float(sample.frame_index))
            y_values.append(value)
            self.buffer.put(TwoDimPointData(x=float(sample.frame_index), y=value))
        self.buffer.close()
        return TwoDimGraphData(x=x_values, y=y_values, title="Streaming")


class StreamingTextVisualizer(IVisualizer):
    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        points = list(data)
        return (
            TextArtifact(
                kind="text",
                title="Streaming text",
                content=f"streamed={len(points)};sum={sum(point.y for point in points)}",
            ),
        )


def _frame(index: int, value: int) -> Frame:
    image = np.full((4, 4, 3), value, dtype=np.uint8)
    return Frame(image=image, index=index, timestamp_seconds=index / 10.0)
