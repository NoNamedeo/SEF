from __future__ import annotations

import numpy as np

import sef
from examples.minimal_pipeline import (
    DemoFrameExtractor,
    DemoSignalExtractor,
    SampleCountAnalyzer,
    SummaryVisualizer,
    build_registry,
)
from library.core.artifacts import BoxSignalSample, Frame, FrameBuffer, Signal, TwoDimGraphData
from library.core.visualization import TextArtifact


def test_pipeline_facade_runs_registered_plugin_names() -> None:
    outputs = (
        sef.pipeline("quickstart", registry=build_registry())
        .frames("demo_frames", frame_count=3)
        .signals("demo_signals")
        .analyze("sample_count")
        .visualize("summary_text")
        .run()
    )

    assert outputs.results[0].y == [3.0]
    assert outputs.final_artifacts[0].content == "Sample count: 3.0"


def test_pipeline_facade_auto_registers_component_classes() -> None:
    outputs = (
        sef.pipeline("class-components", include_builtins=False)
        .frames(DemoFrameExtractor, frame_count=4)
        .signals(DemoSignalExtractor)
        .analyze(SampleCountAnalyzer)
        .visualize(SummaryVisualizer)
        .run()
    )

    assert outputs.results[0].y == [4.0]
    assert outputs.final_artifacts[0].content == "Sample count: 4.0"


def test_pipeline_facade_accepts_component_instances() -> None:
    outputs = (
        sef.pipeline("instance-components", include_builtins=False)
        .frames(DemoFrameExtractor(frame_count=2))
        .signals(DemoSignalExtractor())
        .analyze(SampleCountAnalyzer())
        .visualize(SummaryVisualizer())
        .run()
    )

    assert outputs.results[0].y == [2.0]
    assert outputs.final_artifacts[0].content == "Sample count: 2.0"


def test_pipeline_facade_accepts_plain_processor_functions() -> None:
    def brighten(image, amount: int = 1):
        return image + amount

    outputs = (
        sef.pipeline("function-processor", include_builtins=False)
        .frames(DemoFrameExtractor, frame_count=3)
        .process(brighten, amount=2)
        .signals(DemoSignalExtractor)
        .analyze(SampleCountAnalyzer)
        .visualize(SummaryVisualizer)
        .run()
    )

    assert outputs.results[0].y == [3.0]


@sef.frame_extractor("decorated_test_frames")
def decorated_frames(frame_count: int = 3) -> FrameBuffer:
    buffer = FrameBuffer(frame_count)
    for index in range(frame_count):
        buffer.put(
            Frame(
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                index=index,
                timestamp_seconds=float(index),
            )
        )
    buffer.close()
    return buffer


@sef.signal_extractor("decorated_test_signals")
def decorated_signals(buffer: FrameBuffer) -> Signal:
    return Signal(
        [
            BoxSignalSample(
                frame_index=frame.index or 0,
                box=(0, 0, 2, 2),
                centroid=(1.0, float(frame.index or 0)),
                timestamp_seconds=frame.timestamp_seconds,
            )
            for frame in buffer
        ]
    )


@sef.analyzer("decorated_test_count")
def decorated_count(signal: Signal) -> TwoDimGraphData:
    return TwoDimGraphData(x=[0.0], y=[float(len(list(signal)))], title="Decorated count")


@sef.visualizer("decorated_test_summary")
def decorated_summary(data: TwoDimGraphData):
    return TextArtifact(
        kind="text",
        title="Decorated summary",
        content=f"Decorated count: {data.y[0]}",
    )


def test_decorators_register_function_plugins_for_default_facade_registry() -> None:
    outputs = (
        sef.pipeline("decorated-functions", include_builtins=False)
        .frames("decorated_test_frames", frame_count=5)
        .signals("decorated_test_signals")
        .analyze("decorated_test_count")
        .visualize("decorated_test_summary")
        .run()
    )

    assert outputs.results[0].y == [5.0]
    assert outputs.final_artifacts[0].content == "Decorated count: 5.0"
