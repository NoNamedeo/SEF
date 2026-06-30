from __future__ import annotations

import time

import numpy as np
import pytest

import sef
from examples.minimal_pipeline import (
    DemoFrameExtractor,
    DemoSignalExtractor,
    SampleCountAnalyzer,
    SummaryVisualizer,
    build_registry,
)
from sef.core import Event, IBranchingRule, IEventEmitter, PluginCategory, PluginRegistry
from sef.core.artifacts import Frame, Signal
from sef.core.artifacts.buffer import FrameBuffer
from sef.core.artifacts.data import TwoDimGraphData
from sef.core.artifacts.signal_sample import BoxSignalSample
from sef.core.events import PipelineLifecycleEvent
from sef.core.interfaces import StageCapabilities
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


def test_top_level_run_accepts_pipeline_facade_and_metadata() -> None:
    pipeline = (
        sef.pipeline("top-level-run", registry=build_registry())
        .frames("demo_frames", frame_count=2)
        .signals("demo_signals")
        .analyze("sample_count")
    )

    outputs = sef.run(pipeline, metadata={"owner": "matteo"})

    assert outputs.metadata.pipeline_id == "top-level-run"
    assert outputs.results[0].y == [2.0]
    assert outputs.metadata.execution_metadata["owner"] == "matteo"


def test_top_level_submit_accepts_pipeline_facade() -> None:
    pipeline = (
        sef.pipeline("top-level-submit", registry=build_registry())
        .frames("demo_frames", frame_count=2)
        .signals("demo_signals")
        .analyze("sample_count")
    )

    future = sef.submit(pipeline)
    outputs = future.result(timeout=5)

    assert outputs.metadata.pipeline_id == "top-level-submit"
    assert outputs.results[0].y == [2.0]


def test_top_level_run_accepts_run_config_schema() -> None:
    config = {
        "schema_version": "1.0",
        "id": "config-run",
        "metadata": {"owner": "matteo"},
        "run": {
            "execution_plan": "summary",
            "runtime": {"frame_buffer_size": 4},
        },
        "pipeline": {
            "frame_extractor": {"name": "demo_frames", "params": {"frame_count": 2}},
            "signal_extractor": {"name": "demo_signals"},
            "analyzers": [{"name": "sample_count"}],
        },
    }

    outputs = sef.run(config, registry=build_registry())

    assert outputs.metadata.pipeline_id == "config-run"
    assert outputs.metadata.execution_metadata["owner"] == "matteo"
    assert outputs.metadata.execution_plan["stage_count"] == 3


def test_top_level_run_rejects_pipeline_context() -> None:
    context = (
        sef.pipeline("context-is-core-only", registry=build_registry())
        .frames("demo_frames", frame_count=1)
        .signals("demo_signals")
        .analyze("sample_count")
        .build_context()
    )

    with pytest.raises(TypeError, match="PipelineContext"):
        sef.run(context)


def test_pipeline_to_config_emits_run_document_runtime() -> None:
    config = (
        sef.pipeline("run-document", registry=build_registry())
        .frames("demo_frames", frame_count=2)
        .signals("demo_signals")
        .analyze("sample_count")
        .runtime(frame_buffer_size=4)
        .to_config(metadata={"owner": "matteo"}, run={"execution_plan": "summary"})
    )

    assert config["id"] == "run-document"
    assert config["metadata"] == {"owner": "matteo"}
    assert config["run"]["execution_plan"] == "summary"
    assert config["run"]["runtime"]["frame_buffer_size"] == 4
    assert "runtime" not in config["pipeline"]


def test_from_config_preserves_run_section() -> None:
    outputs = sef.from_config(
        {
            "schema_version": "1.0",
            "run": {
                "execution_plan": "summary",
                "reproducibility": True,
            },
            "pipeline": {
                "frame_extractor": {"name": "demo_frames", "params": {"frame_count": 2}},
                "signal_extractor": {"name": "demo_signals"},
                "analyzers": [{"name": "sample_count"}],
            },
        },
        registry=build_registry(),
    ).run()

    assert outputs.metadata.execution_plan["stage_count"] == 3
    assert "stages" not in outputs.metadata.execution_plan
    assert outputs.metadata.reproducibility["config"]["run"]["execution_plan"] == "summary"
    assert outputs.metadata.reproducibility["config"]["run"]["reproducibility"] is True


def test_explicit_run_overrides_configured_run() -> None:
    facade = sef.from_config(
        {
            "schema_version": "1.0",
            "run": {
                "execution_plan": "full",
                "reproducibility": True,
            },
            "pipeline": {
                "frame_extractor": {"name": "demo_frames", "params": {"frame_count": 2}},
                "signal_extractor": {"name": "demo_signals"},
                "analyzers": [{"name": "sample_count"}],
            },
        },
        registry=build_registry(),
    )

    outputs = facade.run(run={"execution_plan": "none", "reproducibility": False})

    assert outputs.metadata.execution_plan == {}
    assert outputs.metadata.reproducibility == {}


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


def test_pipeline_facade_accepts_plain_frame_buffer_processor_functions() -> None:
    def mark_buffer(buffer: FrameBuffer, label: str) -> FrameBuffer:
        output = buffer.clone_empty()
        for frame in buffer:
            output.put(
                Frame(
                    image=frame.image,
                    index=frame.index,
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata={**frame.metadata, "label": label},
                )
            )
        output.close()
        return output

    outputs = (
        sef.pipeline("function-buffer-processor", include_builtins=False)
        .frames(DemoFrameExtractor, frame_count=3)
        .process(mark_buffer, processor_type="frame_buffer", label="processed")
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


@sef.analyzer
def decorated_test_legacy_count(signal: Signal) -> TwoDimGraphData:
    return TwoDimGraphData(x=[0.0], y=[float(len(list(signal)))], title="Legacy decorated count")


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


def test_decorator_without_parentheses_remains_supported() -> None:
    registry = sef.default_registry(include_builtins=False)
    definition = registry.get(PluginCategory.ANALYZER, "decorated_test_legacy_count")

    assert definition.name == "decorated_test_legacy_count"
    assert definition.metadata["source"] == "sef.decorator"


def test_decorators_register_rich_plugin_metadata_aliases_and_capabilities() -> None:
    registry = PluginRegistry()
    capabilities = StageCapabilities.streaming(stateful=False, realtime_safe=True)

    @sef.analyzer(
        "rich_count",
        registry=registry,
        description="Count samples with a configurable scale.",
        version="2.0.0",
        aliases=("count_alias",),
        metadata={
            "domain": "tests",
            "tags": ["decorator", "metadata"],
            "params": {"scale": {"type": "float", "default": 1.0}},
        },
        capabilities=capabilities,
    )
    def rich_count(signal: Signal, scale: float = 1.0) -> TwoDimGraphData:
        return TwoDimGraphData(x=[0.0], y=[float(len(list(signal))) * scale], title="Rich count")

    definition = registry.get(PluginCategory.ANALYZER, "count_alias")
    assert definition.name == "rich_count"
    assert definition.description == "Count samples with a configurable scale."
    assert definition.version == "2.0.0"
    assert definition.aliases == ("count_alias",)
    assert definition.metadata["domain"] == "tests"
    assert definition.metadata["source"] == "sef.decorator"
    assert definition.factory.capabilities == capabilities

    analyzer = registry.create(PluginCategory.ANALYZER, "rich_count", scale=2.0)
    assert analyzer.capabilities == capabilities


def test_frame_buffer_processor_decorator_registers_buffer_level_functions() -> None:
    registry = PluginRegistry()
    capabilities = StageCapabilities.batch(stateful=False, realtime_safe=False)

    @sef.frame_buffer_processor(
        "buffer_passthrough",
        registry=registry,
        description="Return frames unchanged.",
        metadata={"domain": "frames"},
        capabilities=capabilities,
    )
    def buffer_passthrough(buffer: FrameBuffer) -> FrameBuffer:
        return buffer

    definition = registry.get(PluginCategory.FRAME_BUFFER_PROCESSOR, "buffer_passthrough")
    assert definition.description == "Return frames unchanged."
    assert definition.metadata["domain"] == "frames"
    assert definition.factory.capabilities == capabilities

    processor = registry.create(PluginCategory.FRAME_BUFFER_PROCESSOR, "buffer_passthrough")
    assert processor.capabilities == capabilities


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

    future = sef.orchestrator().submit_context(context, id="submitted-context")
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

    def build_config(self, event: Event):
        return (
            sef.pipeline("child-from-event", registry=build_registry(), include_builtins=False)
            .frames("demo_frames", frame_count=int(event.require("sample_count")))
            .signals("demo_signals")
            .analyze("sample_count")
            .visualize("summary_text")
            .to_config()
        )


def test_orchestrator_facade_wires_branching_with_run_config_schema() -> None:
    orchestrator = sef.orchestrator(registry=build_registry(), include_builtins=False)
    orchestrator.with_branching(DemoBranchRule()).with_branching(DemoBranchRule())
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
