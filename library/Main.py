from __future__ import annotations

import time

import numpy as np

from library.core.artifacts.BoxSignalSample import BoxSignalSample
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.events.Event import Event
from library.core.events.EventBus import EventBus
from library.core.events.PipelineEvent import PipelineEvent
from library.core.events.PipelineLifecycleEvent import PipelineLifecycleEvent
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry

# ---------------------------------------------------------------------------
# Minimal in-memory components used by the examples.
# They keep Main.py runnable without videos, OpenCV windows, or filesystem data.
# ---------------------------------------------------------------------------


class SyntheticFrameExtractor(IFrameExtractor):
    """Creates a deterministic in-memory frame sequence."""

    def __init__(self, count: int = 5) -> None:
        super().__init__()
        self._count = count

    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(self._count)
        for index in range(self._count):
            image = np.full((4, 4, 3), fill_value=index, dtype=np.uint8)
            buffer.put(Frame(image=image, index=index, timestamp_seconds=index * 0.1))
        buffer.close()
        return buffer


class MetadataFrameCleaner(IFrameCleaner):
    """Example frame-cleaning stage: returns frames with extra metadata."""

    def clean(self, frame: Frame) -> Frame:
        metadata = {**frame.metadata, "cleaned": True}
        return Frame(
            image=frame.image,
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata=metadata,
        )


class SyntheticSignalExtractor(ISignalExtractor, IEventEmitter):
    """Converts frames into centroid samples and emits a domain event."""

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples: list[BoxSignalSample] = []
        for frame in buffer:
            frame_index = int(frame.index or 0)
            samples.append(
                BoxSignalSample(
                    frame_index=frame_index,
                    box=(0, 0, 4, 4),
                    centroid=(float(frame_index), float(frame_index * 2)),
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata=frame.metadata,
                )
            )
        self.emit("example.signal_ready", {"samples": len(samples)})
        return Signal(samples)


class ScaleSignalCleaner(ISignalCleaner):
    """Example signal cleaner: scales the y coordinate of each centroid."""

    def __init__(self, factor: float = 1.0) -> None:
        super().__init__()
        self._factor = factor

    def clean(self, signal: ISignal) -> ISignal:
        scaled = []
        for sample in signal:
            centroid = sample.centroid
            scaled.append(
                BoxSignalSample(
                    frame_index=sample.frame_index,
                    box=sample.box,
                    centroid=(
                        None
                        if centroid is None
                        else (centroid[0], centroid[1] * self._factor)
                    ),
                    timestamp_seconds=sample.timestamp_seconds,
                    metadata=sample.metadata,
                )
            )
        return Signal(scaled)


class PositionAnalyzer(IAnalyzer):
    """Returns the y position series."""

    def analyze(self, signal: ISignal) -> IData:
        x = [float(sample.frame_index) for sample in signal]
        y = [float(sample.centroid[1]) for sample in signal if sample.centroid]
        return TwoDimGraphData(x=x, y=y, label="position", title="Position")


class VelocityAnalyzer(IAnalyzer):
    """Returns a simple frame-to-frame delta series."""

    def analyze(self, signal: ISignal) -> IData:
        values = [float(sample.centroid[1]) for sample in signal if sample.centroid]
        deltas = [
            0.0,
            *[current - previous for previous, current in zip(values, values[1:])],
        ]
        return TwoDimGraphData(
            x=[float(index) for index in range(len(deltas))],
            y=deltas,
            label="velocity",
            title="Velocity",
        )


class ConsoleVisualizer(IVisualizer):
    """Small visualizer that prints which analysis result it received."""

    def visualize(self, data: IData) -> None:
        if isinstance(data, TwoDimGraphData):
            print(f"visualizer received {data.label}: {data.y}")


# ---------------------------------------------------------------------------
# Context factories.
# ---------------------------------------------------------------------------


def build_direct_context() -> PipelineContext:
    """Direct construction: fastest way when dependencies are already objects."""
    return PipelineContext(
        frame_extractor=SyntheticFrameExtractor(count=4),
        frame_cleaners=[MetadataFrameCleaner()],
        signal_extractor=SyntheticSignalExtractor(),
        signal_cleaners=[ScaleSignalCleaner(factor=1.5)],
        analyzers=[PositionAnalyzer()],
    )


def build_fluent_context() -> PipelineContext:
    """Programmatic composition via FluentPipelineBuilder."""
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(SyntheticFrameExtractor(count=4))
        .add_frame_cleaner(MetadataFrameCleaner())
        .with_signal_extractor(SyntheticSignalExtractor())
        .add_signal_cleaner(ScaleSignalCleaner(factor=2.0))
        .with_analyzers([PositionAnalyzer(), VelocityAnalyzer()])
        .add_visualizer_for_results(ConsoleVisualizer(), [1])
        .build_context()
    )


def build_example_registry() -> PluginRegistry:
    """Registry used by ConfigPipelineBuilder examples."""
    registry = PluginRegistry()
    registry.register(PluginCategory.FRAME_EXTRACTOR, "synthetic_frames", SyntheticFrameExtractor)
    registry.register(PluginCategory.FRAME_CLEANER, "metadata_cleaner", MetadataFrameCleaner)
    registry.register(PluginCategory.SIGNAL_EXTRACTOR, "synthetic_signal", SyntheticSignalExtractor)
    registry.register(PluginCategory.SIGNAL_CLEANER, "scale_signal", ScaleSignalCleaner)
    registry.register(PluginCategory.ANALYZER, "position", PositionAnalyzer)
    registry.register(PluginCategory.ANALYZER, "velocity", VelocityAnalyzer)
    registry.register(PluginCategory.VISUALIZER, "console", ConsoleVisualizer)
    return registry


def build_config_context() -> PipelineContext:
    """Declarative composition: useful for YAML/JSON/UI-driven pipelines."""
    config = {
        "pipeline": {
            "frame_extractor": {"name": "synthetic_frames", "params": {"count": 4}},
            "frame_cleaners": [{"name": "metadata_cleaner"}],
            "signal_extractor": {"name": "synthetic_signal"},
            "signal_cleaners": [{"name": "scale_signal", "params": {"factor": 2.0}}],
            "analyzers": [{"name": "position"}, {"name": "velocity"}],
            "visualizers": [{"name": "console", "result_indices": [0]}],
        }
    }
    return ConfigPipelineBuilder(build_example_registry()).build_context(config)


# ---------------------------------------------------------------------------
# Usage examples.
# ---------------------------------------------------------------------------


def example_sync_run() -> None:
    """Basic synchronous execution through the application facade."""
    orchestrator = PipelineOrchestrator()
    try:
        print_example_header("sync run")
        results = orchestrator.run(build_direct_context(), pipeline_id="sync-demo")
        print_result_summary(results)
    finally:
        orchestrator.shutdown()


def example_fluent_builder() -> None:
    """Fluent builder with multiple analyzers and a selective visualizer."""
    orchestrator = PipelineOrchestrator()
    try:
        print_example_header("fluent builder")
        results = orchestrator.run(build_fluent_context(), pipeline_id="fluent-demo")
        print_result_summary(results)
    finally:
        orchestrator.shutdown()


def example_config_builder() -> None:
    """Config-driven build using PluginRegistry and ConfigPipelineBuilder."""
    orchestrator = PipelineOrchestrator()
    try:
        print_example_header("config builder")
        results = orchestrator.run(build_config_context(), pipeline_id="config-demo")
        print_result_summary(results)
    finally:
        orchestrator.shutdown()


def example_async_multi_pipeline() -> None:
    """Run multiple pipelines concurrently and inspect rich run snapshots."""
    monitor = InMemoryPipelineMonitor()
    runner = ThreadedPipelineRunner(monitor=monitor, max_workers=2)
    orchestrator = PipelineOrchestrator(runner=runner)
    try:
        print_example_header("async multi-pipeline")
        orchestrator.submit(build_direct_context(), pipeline_id="async-a")
        orchestrator.submit(build_fluent_context(), pipeline_id="async-b")
        wait_until_idle(runner)
        for snapshot in runner.snapshots():
            print(
                f"snapshot {snapshot.pipeline_id}: "
                f"state={snapshot.state}, attempt={snapshot.attempt}, error={snapshot.error}"
            )
    finally:
        runner.shutdown()


def example_event_driven_trigger_and_lifecycle() -> None:
    """Trigger execution via EventBus and observe lifecycle/domain events."""
    trigger_bus = EventBus()
    domain_bus = EventBus()
    runner = ThreadedPipelineRunner(
        monitor=InMemoryPipelineMonitor(),
        lifecycle_bus=trigger_bus,
    )
    orchestrator = PipelineOrchestrator(
        runner=runner,
        bus=trigger_bus,
        domain_bus=domain_bus,
    )

    trigger_bus.subscribe(PipelineLifecycleEvent.BEFORE_RUN, print_event)
    trigger_bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, print_event)
    domain_bus.subscribe("example.signal_ready", print_event)

    try:
        print_example_header("event-driven trigger and lifecycle")
        trigger_bus.dispatch(
            PipelineEvent.create(
                pipeline_id="event-demo",
                context=build_direct_context(),
                source="Main.example_event_driven_trigger_and_lifecycle",
            )
        )
        wait_until_idle(runner)
    finally:
        runner.shutdown()


def example_visualizer_targeting() -> None:
    """
    Visualizer targeting: one visualizer may receive all results by default,
    or only selected analyzer outputs through result_indices.
    """
    context = build_fluent_context()
    orchestrator = PipelineOrchestrator()
    try:
        print_example_header("visualizer targeting")
        orchestrator.run(context, pipeline_id="visualizer-target-demo")
    finally:
        orchestrator.shutdown()


def print_example_header(title: str) -> None:
    print(f"\n[{title}]")


def print_result_summary(results: list[IData]) -> None:
    for result in results:
        if isinstance(result, TwoDimGraphData):
            print(f"- {result.label}: {result.y}")
        else:
            print(f"- {type(result).__name__}")


def print_event(event: Event) -> None:
    pipeline_id = event.payload.get("pipeline_id", "-")
    print(f"event {event.event_type} from {event.source} pipeline={pipeline_id}")


def wait_until_idle(runner: ThreadedPipelineRunner, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while runner.active_ids() and time.monotonic() < deadline:
        time.sleep(0.01)


def main() -> None:
    """Run a compact tour of the core usage modes."""
    example_sync_run()
    example_fluent_builder()
    example_config_builder()
    example_async_multi_pipeline()
    example_event_driven_trigger_and_lifecycle()
    example_visualizer_targeting()


if __name__ == "__main__":
    main()
