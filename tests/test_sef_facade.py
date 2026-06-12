from __future__ import annotations

import time

import numpy as np

import sef
from examples.minimal_pipeline import (
    DemoFrameExtractor,
    DemoSignalExtractor,
    SampleCountAnalyzer,
    SummaryVisualizer,
    build_registry,
)
from sef.core import Event, IBranchingRule, IEventEmitter
from sef.core.artifacts import Frame, Signal
from sef.core.artifacts.buffer import FrameBuffer
from sef.core.artifacts.data import TwoDimGraphData
from sef.core.artifacts.signal_sample import BoxSignalSample
from sef.core.events import PipelineLifecycleEvent
from sef.core.visualization import TextArtifact


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


def test_orchestrator_facade_runs_pipeline_facade_and_emits_lifecycle_events() -> None:
    events: list[Event] = []
    pipeline = (
        sef.pipeline("orchestrated-run", include_builtins=False)
        .frames(DemoFrameExtractor, frame_count=2)
        .signals(DemoSignalExtractor)
        .analyze(SampleCountAnalyzer)
        .visualize(SummaryVisualizer)
    )

    outputs = sef.orchestrator().on_lifecycle("after_run", events.append).run(pipeline)

    assert outputs.results[0].y == [2.0]
    assert len(events) == 1
    assert events[0].event_type == PipelineLifecycleEvent.AFTER_RUN
    assert events[0].payload["pipeline_id"] == "orchestrated-run"


def test_orchestrator_facade_submits_pipeline_context() -> None:
    context = (
        sef.pipeline("submitted-context", include_builtins=False)
        .frames(DemoFrameExtractor, frame_count=2)
        .signals(DemoSignalExtractor)
        .analyze(SampleCountAnalyzer)
        .visualize(SummaryVisualizer)
        .build_context()
    )

    future = sef.orchestrator().submit(context, pipeline_id="submitted-context")
    outputs = future.result(timeout=5)

    assert outputs.results[0].y == [2.0]


class EmittingDemoSignalExtractor(DemoSignalExtractor, IEventEmitter):
    """Demo extractor that emits a branchable domain event."""

    def extract(self, buffer: FrameBuffer) -> Signal:
        signal = super().extract(buffer)
        self.emit("demo.branch", {"sample_count": len(signal)})
        return signal


class DemoBranchRule(IBranchingRule):
    """Spawn one deterministic child pipeline from a demo domain event."""

    def matches(self, event: Event) -> bool:
        return event.event_type == "demo.branch"

    def build_context(self, event: Event):
        return (
            sef.pipeline("child-from-event", include_builtins=False)
            .frames(DemoFrameExtractor, frame_count=int(event.require("sample_count")))
            .signals(DemoSignalExtractor)
            .analyze(SampleCountAnalyzer)
            .visualize(SummaryVisualizer)
            .build_context()
        )


def test_orchestrator_facade_wires_branching_without_config_schema() -> None:
    orchestrator = sef.orchestrator().with_branching(DemoBranchRule()).with_branching(DemoBranchRule())
    events: list[Event] = []
    orchestrator.on_lifecycle("after_run", events.append)
    pipeline = (
        sef.pipeline("primary", include_builtins=False)
        .frames(DemoFrameExtractor, frame_count=3)
        .signals(EmittingDemoSignalExtractor)
        .analyze(SampleCountAnalyzer)
        .visualize(SummaryVisualizer)
    )

    future = orchestrator.submit(pipeline)
    future.result(timeout=5)
    _wait_until_idle(orchestrator)
    orchestrator.shutdown()

    pipeline_ids = {event.payload["pipeline_id"] for event in events}
    secondary_ids = {pipeline_id for pipeline_id in pipeline_ids if pipeline_id.startswith("secondary-")}
    assert "primary" in pipeline_ids
    assert len(secondary_ids) == 2


def _wait_until_idle(orchestrator, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while orchestrator.active_ids() and time.monotonic() < deadline:
        time.sleep(0.01)
