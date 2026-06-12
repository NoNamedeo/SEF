from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from sef.builtin.analyzers.NoAnalyzer import NoAnalyzer
from sef.core.artifacts.signal_sample.BoxSignalSample import BoxSignalSample
from sef.core.artifacts.buffer.DataBuffer import DataBuffer
from sef.core.artifacts.Frame import Frame
from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.artifacts.Signal import Signal
from sef.core.artifacts.buffer.SignalBuffer import SignalBuffer
from sef.core.artifacts.data.TwoDimGraphData import TwoDimGraphData
from sef.core.artifacts.data.TwoDimPointData import TwoDimPointData
from sef.core.interfaces.BufferContracts import (
    IBuffer,
    IBufferSubscription,
    IFrameBuffer,
    ISubscribableBuffer,
)
from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.IData import IData
from sef.core.interfaces.IFrameBufferProcessor import IFrameBufferProcessor
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalSample import ISignalSample
from sef.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from sef.core.interfaces.StageCapabilities import StageCapabilities
from sef.core.interfaces.StreamingContracts import (
    IStreamingAnalyzer,
    IStreamingFrameExtractor,
    IStreamingSignalExtractor,
    IStreamingVisualizer,
)
from sef.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from sef.core.pipeline.LatencyPolicy import (
    AdaptiveSamplingFrameLatencyPolicy,
    DropNewestFrameLatencyPolicy,
    FrameLatencyPolicy,
)
from sef.core.pipeline.Pipeline import Pipeline
from sef.core.pipeline.PipelineExecutionPolicy import (
    DefaultPipelineExecutionPolicy,
    PipelineExecutionDecision,
    PipelineExecutionMode,
    PipelineStagePolicyContext,
)
from sef.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from sef.core.visualization.VisualArtifact import TextArtifact, VideoFileArtifact, VisualArtifact
from sef.core.visualization.VisualizationContext import VisualizationContext
from sef.builtin.exporters.OpenCVFrameBufferVideoExporter import OpenCVFrameBufferVideoExporter
from sef.builtin.signal_extractors.NoSignalExtractor import NoSignalExtractor


def test_frame_buffer_can_close_when_full() -> None:
    buffer = FrameBuffer(buffer_size=2)
    buffer.put(_frame(0, 0))
    buffer.put(_frame(1, 1))

    buffer.close()

    assert [frame.index for frame in buffer] == [0, 1]


def test_frame_buffer_rejects_new_frames_after_abort() -> None:
    buffer = FrameBuffer(buffer_size=1)
    buffer.abort()

    buffer.put(_frame(0, 0))

    assert buffer.try_put(_frame(1, 1)) is False
    assert list(buffer) == []


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


def test_buffers_satisfy_runtime_contracts() -> None:
    frame_buffer = FrameBuffer(buffer_size=1)
    signal_buffer = SignalBuffer(buffer_size=1)
    data_buffer = DataBuffer(buffer_size=1)

    assert isinstance(frame_buffer, IBuffer)
    assert isinstance(frame_buffer, IFrameBuffer)
    assert isinstance(signal_buffer, ISubscribableBuffer)
    assert isinstance(data_buffer, ISubscribableBuffer)

    signal_buffer.set_consumer_count(1)
    assert isinstance(signal_buffer.subscribe(0), IBufferSubscription)


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


def test_pipeline_runs_main_thread_streaming_visualizer_on_calling_thread() -> None:
    visualizer = MainThreadStreamingTextVisualizer()
    expected_thread_id = threading.get_ident()
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StreamingFrameExtractor(frame_count=3))
        .with_signal_extractor(StreamingSignalExtractor())
        .add_analyzer(StreamingAnalyzer())
        .add_visualizer_for_results(visualizer, [0])
        .build_context()
    )

    outputs = Pipeline(context).run()

    assert visualizer.render_thread_id == expected_thread_id
    assert len(outputs.final_artifacts) == 1
    assert isinstance(outputs.final_artifacts[0], TextArtifact)
    assert outputs.final_artifacts[0].content == "streamed=3;sum=3.0"


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


def test_pipeline_streams_prefix_before_sequence_processor_without_deadlock(tmp_path: Path) -> None:
    output_path = tmp_path / "hybrid.mp4"
    add_one = AddOneProcessor()
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StreamingFrameExtractor(frame_count=8))
        .add_frame_processor(SingleFrameProcessorAdapter(add_one))
        .add_frame_processor(SequenceMeanFrameProcessor())
        .add_frame_exporter(OpenCVFrameBufferVideoExporter(output_path, fps=10.0, max_exported_frames=8))
        .with_signal_extractor(NoSignalExtractor())
        .add_analyzer(NoAnalyzer())
        .build_context()
    )

    outputs = Pipeline(context).run()

    assert add_one.processed_indexes == tuple(range(8))
    assert len(outputs.final_artifacts) == 1
    assert isinstance(outputs.final_artifacts[0], VideoFileArtifact)
    assert outputs.final_artifacts[0].metadata["frame_count"] == 8
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_execution_plan_reports_streaming_and_materialization_boundary(tmp_path: Path) -> None:
    output_path = tmp_path / "planned.mp4"
    context = (
        FluentPipelineBuilder()
        .with_stream_runtime(
            {
                "frame_buffer_size": 2,
                "latency_policy": {"name": "drop_newest", "params": {}},
            }
        )
        .with_frame_extractor(StreamingFrameExtractor(frame_count=5))
        .add_frame_processor(SingleFrameProcessorAdapter(AddOneProcessor()))
        .add_frame_processor(SequenceMeanFrameProcessor())
        .add_frame_exporter(OpenCVFrameBufferVideoExporter(output_path, fps=10.0, max_exported_frames=5))
        .with_signal_extractor(NoSignalExtractor())
        .add_analyzer(NoAnalyzer())
        .build_context()
    )

    pipeline = Pipeline(context)
    plan = pipeline.execution_plan()

    assert not plan.streamable_end_to_end
    assert [stage.stage_id for stage in plan.materialization_boundaries] == ["frame_processing[1]"]
    assert plan.runtime["latency_policy"]["name"] == "drop_newest"

    outputs = pipeline.run()

    execution_plan = outputs.metadata.execution_plan
    assert execution_plan["runtime"]["latency_policy"]["name"] == "drop_newest"
    assert execution_plan["materialization_boundaries"][0]["stage_id"] == "frame_processing[1]"


def test_pipeline_avoids_isolated_streaming_switch_before_batch_processor() -> None:
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StreamingFrameExtractor(frame_count=4))
        .add_frame_processor(SequenceMeanFrameProcessor())
        .with_signal_extractor(NoSignalExtractor())
        .add_analyzer(NoAnalyzer())
        .build_context()
    )

    plan = Pipeline(context).execution_plan()
    stages = {stage.stage_id: stage for stage in plan.stages}

    assert stages["frame_extraction"].execution_mode == "batch"
    assert stages["frame_processing[0]"].execution_mode == "batch"
    assert plan.materialization_boundaries == ()


def test_pipeline_restarts_streaming_after_frame_batch_boundary(tmp_path: Path) -> None:
    output_path = tmp_path / "restart.mp4"
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StreamingFrameExtractor(frame_count=5))
        .add_frame_processor(SingleFrameProcessorAdapter(AddOneProcessor()))
        .add_frame_processor(SequenceMeanFrameProcessor())
        .add_frame_exporter(OpenCVFrameBufferVideoExporter(output_path, fps=10.0, max_exported_frames=5))
        .with_signal_extractor(StreamingSignalExtractor())
        .add_analyzer(StreamingAnalyzer())
        .add_visualizer_for_results(StreamingTextVisualizer(), [0])
        .build_context()
    )

    pipeline = Pipeline(context)
    plan = pipeline.execution_plan()
    stages = {stage.stage_id: stage for stage in plan.stages}

    assert stages["frame_processing[1]"].materializes_input
    assert stages["frame_export[0]"].execution_mode == "streaming"
    assert stages["signal_extraction"].execution_mode == "streaming"
    assert stages["analysis[0]"].execution_mode == "streaming"

    outputs = pipeline.run()

    video_artifacts = [artifact for artifact in outputs.final_artifacts if isinstance(artifact, VideoFileArtifact)]
    text_artifacts = [artifact for artifact in outputs.final_artifacts if isinstance(artifact, TextArtifact)]
    assert len(video_artifacts) == 1
    assert len(text_artifacts) == 1
    assert text_artifacts[0].content == "streamed=5;sum=15.0"


def test_pipeline_runs_streaming_and_batch_analyzers_from_same_signal_stream() -> None:
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StreamingFrameExtractor(frame_count=3))
        .with_signal_extractor(StreamingSignalExtractor())
        .add_analyzer(StreamingAnalyzer())
        .add_analyzer(BatchSumAnalyzer())
        .add_visualizer_for_results(StreamingTextVisualizer(), [0])
        .build_context()
    )

    pipeline = Pipeline(context)
    plan = pipeline.execution_plan()
    stages = {stage.stage_id: stage for stage in plan.stages}

    assert stages["analysis[0]"].execution_mode == "streaming"
    assert stages["analysis[1]"].execution_mode == "batch"
    assert stages["analysis[1]"].materializes_input

    outputs = pipeline.run()

    assert len(outputs.results) == 2
    assert outputs.results[0].y == [0.0, 1.0, 2.0]
    assert outputs.results[1].y == [3.0]
    assert len(outputs.final_artifacts) == 1
    assert outputs.final_artifacts[0].content == "streamed=3;sum=3.0"


def test_pipeline_uses_interchangeable_execution_policy() -> None:
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StreamingFrameExtractor(frame_count=4))
        .add_frame_processor(SequenceMeanFrameProcessor())
        .with_signal_extractor(NoSignalExtractor())
        .add_analyzer(NoAnalyzer())
        .build_context()
    )

    pipeline = Pipeline(context, execution_policy=ForceStreamingSourcePolicy())
    plan = pipeline.execution_plan()
    stages = {stage.stage_id: stage for stage in plan.stages}

    assert stages["frame_extraction"].execution_mode == "streaming"
    assert stages["frame_extraction"].reason == "test policy forces source streaming"
    assert stages["frame_processing[0]"].materializes_input

    outputs = pipeline.run()

    assert len(outputs.results) == 1


def test_drop_newest_latency_policy_drops_when_queue_is_full() -> None:
    policy = DropNewestFrameLatencyPolicy()
    output = FrameBuffer(buffer_size=1)

    assert policy.publish(_frame(0, 0), output) is True
    assert policy.publish(_frame(1, 1), output) is False

    assert policy.metrics() == {"accepted_frames": 1, "dropped_frames": 1}


def test_adaptive_sampling_policy_reduces_accepted_frames_under_pressure() -> None:
    policy = AdaptiveSamplingFrameLatencyPolicy(
        min_interval=1,
        max_interval=3,
        low_watermark=0.0,
        high_watermark=0.0,
    )
    output = FrameBuffer(buffer_size=2)

    decisions = [policy.publish(_frame(index, index), output) for index in range(6)]

    assert any(not decision for decision in decisions)
    assert policy.metrics()["dropped_frames"] > 0


class StreamingFrameExtractor(IStreamingFrameExtractor):
    capabilities = StageCapabilities.streaming()

    def __init__(self, frame_count: int) -> None:
        super().__init__()
        self._frame_count = frame_count

    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(buffer_size=self._frame_count + 1)
        from sef.core.pipeline.LatencyPolicy import BlockingFrameLatencyPolicy

        self.extract_into(buffer, BlockingFrameLatencyPolicy())
        return buffer

    def extract_into(self, output_buffer: IFrameBuffer, latency_policy: FrameLatencyPolicy) -> None:
        for frame_index in range(self._frame_count):
            latency_policy.publish(_frame(frame_index, frame_index), output_buffer)
        output_buffer.close()


class AddOneProcessor(ISingleFrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.processed_indexes: tuple[int, ...] = ()

    def process(self, frame: Frame) -> Frame:
        self.processed_indexes = (*self.processed_indexes, int(frame.index or 0))
        return Frame(
            image=frame.image + 1,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata=dict(frame.metadata),
        )


class SequenceMeanFrameProcessor(IFrameBufferProcessor):
    capabilities = StageCapabilities.batch()

    def process(self, buffer: FrameBuffer) -> FrameBuffer:
        frames = list(buffer)
        output = FrameBuffer(buffer_size=len(frames) + 1)
        mean_value = int(np.mean([float(frame.image[0, 0, 0]) for frame in frames]))
        for frame in frames:
            output.put(
                Frame(
                    image=np.full_like(frame.image, mean_value),
                    index=frame.index,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata=dict(frame.metadata),
                )
            )
        output.close()
        return output


class StreamingSignalExtractor(IStreamingSignalExtractor):
    capabilities = StageCapabilities.streaming()

    def extract(self, buffer: FrameBuffer) -> ISignal:
        return Signal(list(self._samples(buffer)))

    def extract_into(self, buffer: IFrameBuffer, output_buffer: IBuffer[ISignalSample]) -> None:
        for sample in self._samples(buffer):
            output_buffer.put(sample)
        output_buffer.close()

    @staticmethod
    def _samples(buffer: FrameBuffer):
        for frame in buffer:
            value = float(frame.image[0, 0, 0])
            yield BoxSignalSample(
                frame_index=frame.index or 0,
                box=None,
                centroid=(0.0, value),
                timestamp_seconds=frame.timestamp_seconds,
            )


class StreamingAnalyzer(IStreamingAnalyzer):
    capabilities = StageCapabilities.streaming()

    def analyze(self, signal: ISignal) -> IData:
        return self.analyze_into(signal, DataBuffer(buffer_size=2))

    def analyze_into(self, signal: Iterable[ISignalSample], output_buffer: IBuffer[IData]) -> IData:
        x_values: list[float] = []
        y_values: list[float] = []
        for sample in signal:
            value = float(sample.centroid[1]) if sample.centroid is not None else 0.0
            x_values.append(float(sample.frame_index))
            y_values.append(value)
            output_buffer.put(TwoDimPointData(x=float(sample.frame_index), y=value))
        output_buffer.close()
        return TwoDimGraphData(x=x_values, y=y_values, title="Streaming")


class BatchSumAnalyzer(IAnalyzer):
    capabilities = StageCapabilities.batch()

    def analyze(self, signal: ISignal) -> IData:
        values = [
            float(sample.centroid[1])
            for sample in signal
            if getattr(sample, "centroid", None) is not None
        ]
        return TwoDimGraphData(x=[0.0], y=[sum(values)], title="Batch sum")


class StreamingTextVisualizer(IStreamingVisualizer):
    capabilities = StageCapabilities.streaming()

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        return self.render_stream((data,), context)

    def render_stream(
        self,
        data: Iterable[IData],
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


class MainThreadStreamingTextVisualizer(StreamingTextVisualizer):
    requires_main_thread = True

    def __init__(self) -> None:
        super().__init__()
        self.render_thread_id: int | None = None

    def render_stream(
        self,
        data: Iterable[IData],
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        self.render_thread_id = threading.get_ident()
        return super().render_stream(data, context)


class ForceStreamingSourcePolicy(DefaultPipelineExecutionPolicy):
    def decide_source(self, context: PipelineStagePolicyContext) -> PipelineExecutionDecision:
        if context.stage_streamable:
            return PipelineExecutionDecision(
                PipelineExecutionMode.STREAMING,
                "test policy forces source streaming",
            )
        return super().decide_source(context)


def _frame(index: int, value: int) -> Frame:
    image = np.full((4, 4, 3), value, dtype=np.uint8)
    return Frame(image=image, index=index, timestamp_seconds=index / 10.0)
