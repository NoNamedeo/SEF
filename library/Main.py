from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

from library.core.artifacts.signal_sample.BoxSignalSample import BoxSignalSample
from library.core.artifacts.Frame import Frame
from library.core.artifacts.buffer.FrameBuffer import FrameBuffer
from library.core.artifacts.signal_sample.MultiObjectSignalSample import BoundingBox
from library.core.artifacts.Signal import Signal
from library.core.artifacts.data.TwoDimGraphData import TwoDimGraphData
from library.core.events.Event import Event
from library.core.events.EventBus import EventBus
from library.core.events.PipelineEvent import PipelineEvent
from library.core.events.PipelineLifecycleEvent import PipelineLifecycleEvent
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.ISingleFrameProcessor import ISingleFrameProcessor
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry
from library.core.visualization.PipelineOutputs import PipelineOutputs
from library.core.visualization.VisualArtifact import TextArtifact, VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_extractors.OpenCVMultiObjectSignalExtractor import OpenCVMultiObjectSignalExtractor


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


class MetadataFrameProcessor(ISingleFrameProcessor):
    """Example single-frame processor: returns frames with extra metadata."""

    def process(self, frame: Frame) -> Frame:
        metadata = {**frame.metadata, "processed": True}
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
                    centroid=(None if centroid is None else (centroid[0], centroid[1] * self._factor)),
                    timestamp_seconds=sample.timestamp_seconds,
                    metadata=sample.metadata,
                )
            )
        return Signal(scaled)


class PositionAnalyzer(IAnalyzer):
    """Returns the y position series."""

    def analyze(self, signal: ISignal) -> IData:
        x: list[float] = []
        y: list[float] = []
        for sample in signal:
            centroid = _sample_centroid(sample)
            if centroid is None:
                continue
            x.append(float(sample.frame_index))
            y.append(float(centroid[1]))
        return TwoDimGraphData(x=x, y=y, label="position", title="Position")


class VelocityAnalyzer(IAnalyzer):
    """Returns a simple frame-to-frame delta series."""

    def analyze(self, signal: ISignal) -> IData:
        values: list[float] = []
        for sample in signal:
            centroid = _sample_centroid(sample)
            if centroid is None:
                continue
            values.append(float(centroid[1]))

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

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        if isinstance(data, TwoDimGraphData):
            return (
                TextArtifact(
                    kind="text",
                    title=f"Console summary · {data.label}",
                    content=str(format_series(data.y)),
                ),
            )
        return ()


class SyntheticVideoTracker:
    """
    Deterministic tracker used by the realistic example.

    It keeps the example stable across machines while still exercising the
    real OpenCV frame extraction, frame processing, signal cleaning and analyzer
    pipeline.
    """

    def __init__(
        self,
        horizontal_speed: int = 3,
        vertical_amplitude: int = 14,
        vertical_period: float = 7.0,
    ) -> None:
        self._horizontal_speed = horizontal_speed
        self._vertical_amplitude = vertical_amplitude
        self._vertical_period = vertical_period
        self._start_box = (0, 0, 0, 0)
        self._frame_index = 0

    def init(self, _frame: np.ndarray, box: tuple[int, int, int, int]) -> bool:
        self._start_box = box
        self._frame_index = 0
        return True

    def update(self, _frame: np.ndarray) -> tuple[bool, tuple[int, int, int, int]]:
        self._frame_index += 1
        return True, moving_object_box(
            self._frame_index,
            start_box=self._start_box,
            horizontal_speed=self._horizontal_speed,
            vertical_amplitude=self._vertical_amplitude,
            vertical_period=self._vertical_period,
        )


def _sample_centroid(sample: Any) -> tuple[float, float] | None:
    """
    Resolve a centroid from either single-object or multi-object samples.

    The demo should remain valid whether the extractor returns BoxSignalSample
    or MultiObjectSignalSample. When multiple tracks are present, the primary
    track is the one with the lowest track_id so the output stays deterministic.
    """
    centroid = getattr(sample, "centroid", None)
    if centroid is not None:
        return centroid

    tracks = getattr(sample, "tracks", None)
    if not tracks:
        return None

    ordered_tracks = sorted(
        (track for track in tracks if getattr(track, "centroid", None) is not None),
        key=lambda track: getattr(track, "track_id", 0),
    )
    if not ordered_tracks:
        return None
    return ordered_tracks[0].centroid


# ---------------------------------------------------------------------------
# Context factories.
# ---------------------------------------------------------------------------


def build_direct_context() -> PipelineContext:
    """Direct construction: fastest way when dependencies are already objects."""
    return PipelineContext(
        frame_extractor=SyntheticFrameExtractor(count=4),
        frame_processors=[SingleFrameProcessorAdapter(MetadataFrameProcessor())],
        signal_extractor=SyntheticSignalExtractor(),
        signal_cleaners=[ScaleSignalCleaner(factor=1.5)],
        analyzers=[PositionAnalyzer()],
    )


def build_fluent_context() -> PipelineContext:
    """Programmatic composition via FluentPipelineBuilder."""
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(SyntheticFrameExtractor(count=4))
        .add_frame_processor(SingleFrameProcessorAdapter(MetadataFrameProcessor()))
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
    registry.register(PluginCategory.SINGLE_FRAME_PROCESSOR, "metadata_processor", MetadataFrameProcessor)
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
            "frame_processors": [{"name": "metadata_processor"}],
            "signal_extractor": {"name": "synthetic_signal"},
            "signal_cleaners": [{"name": "scale_signal", "params": {"factor": 2.0}}],
            "analyzers": [{"name": "position"}, {"name": "velocity"}],
            "visualizers": [{"name": "console", "result_indices": [0]}],
        }
    }
    return ConfigPipelineBuilder(build_example_registry()).build_context(config)


def moving_object_box(
    frame_index: int,
    start_box: tuple[int, int, int, int] = (24, 70, 28, 24),
    horizontal_speed: int = 3,
    vertical_amplitude: int = 14,
    vertical_period: float = 7.0,
) -> tuple[int, int, int, int]:
    """Deterministic motion model shared by the video generator and tracker."""
    x, y, width, height = start_box
    vertical_offset = int(vertical_amplitude * np.sin(frame_index / vertical_period))
    return x + frame_index * horizontal_speed, y + vertical_offset, width, height


def create_realistic_demo_video(
    path: Path,
    frame_count: int = 120,
    size: tuple[int, int] = (320, 180),
    fps: float = 24.0,
) -> Path:
    """
    Create a deterministic synthetic traffic-like clip.

    The file is intentionally generated at runtime so the example exercises
    the real OpenCV video path without requiring checked-in media assets.
    """
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create demo video at {path}")

    try:
        width, height = size
        for frame_index in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Background lanes and static visual noise make frame processors meaningful.
            cv2.line(frame, (0, 115), (width, 115), (45, 45, 45), 2)
            cv2.line(frame, (0, 145), (width, 145), (35, 35, 35), 2)
            cv2.rectangle(frame, (210, 35), (248, 58), (35, 65, 110), -1)

            x, y, box_width, box_height = moving_object_box(frame_index)
            cv2.rectangle(
                frame,
                (x, y),
                (x + box_width, y + box_height),
                (40, 210, 255),
                -1,
            )
            cv2.circle(frame, (x + 6, y + box_height), 4, (20, 20, 20), -1)
            cv2.circle(frame, (x + box_width - 6, y + box_height), 4, (20, 20, 20), -1)

            writer.write(frame)
    finally:
        writer.release()

    return path


def select_seed_roi_from_video(
    video_path: str | Path,
    resize: tuple[int, int] | None = None,
    fallback_box: BoundingBox | None = None,
) -> BoundingBox:
    if not _can_show_preview():
        return fallback_box or moving_object_box(0)

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise ValueError("Impossibile leggere il primo frame")

    if resize is not None:
        frame = cv2.resize(frame, resize)

    box = cv2.selectROI("Seleziona croce seed", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Seleziona croce seed")

    x, y, w, h = map(int, box)
    if w <= 0 or h <= 0:
        raise ValueError("ROI non valida")

    return (x, y, w, h)


def build_realistic_sync_context(
    video_path: Path,
    seed_box: BoundingBox | None = None,
    show_preview: bool = True,
) -> PipelineContext:
    """
    Heavier real-world-style composition.

    The ROI is selected on the same resized frame that the extractor will use,
    so the tracker receives coordinates in the correct scale.
    """
    pipeline_resize = (320, 180)
    resolved_seed_box = seed_box or select_seed_roi_from_video(
        video_path,
        resize=pipeline_resize,
        fallback_box=moving_object_box(0),
    )
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                path=video_path,
                config={"resize": pipeline_resize, "stride": 1, "max_frames": 90},
            )
        )
        .with_signal_extractor(
            OpenCVMultiObjectSignalExtractor(
                tracker_type="CSRT",
                start_box=resolved_seed_box,
                max_objects=3,
                template_match_threshold=0.86,
                min_detection_distance=25,
                config={
                    "show": show_preview,
                    "source_path": video_path,
                },
            )
        )
        .with_analyzers([VelocityAnalyzer()])
        .build_context()
    )


# ---------------------------------------------------------------------------
# Usage examples.
# ---------------------------------------------------------------------------


def example_sync_run() -> None:
    """Basic synchronous execution through the application facade."""
    orchestrator = PipelineOrchestrator()
    try:
        print_example_header("sync run")
        outputs = orchestrator.run(build_direct_context(), pipeline_id="sync-demo")
        print_result_summary(outputs)
    finally:
        orchestrator.shutdown()


def example_sync_run_2() -> None:
    """
    Realistic synchronous run with concrete OpenCV components.

    It simulates a heavier application use case: video extraction, multiple
    frame processors, signal extraction, signal smoothing, multiple analyzers
    and selective visualization, all through the orchestrator facade.
    """
    video_path = Path("videos/castello.mp4")
    orchestrator = PipelineOrchestrator()
    try:
        print_example_header("realistic sync run")
        outputs = orchestrator.run(
            build_realistic_sync_context(video_path, show_preview=_can_show_preview()),
            pipeline_id="realistic-sync-demo",
        )
        print_result_summary(outputs)
    finally:
        orchestrator.shutdown()


def example_fluent_builder() -> None:
    """Fluent builder with multiple analyzers and a selective visualizer."""
    orchestrator = PipelineOrchestrator()
    try:
        print_example_header("fluent builder")
        outputs = orchestrator.run(build_fluent_context(), pipeline_id="fluent-demo")
        print_result_summary(outputs)
    finally:
        orchestrator.shutdown()


def example_config_builder() -> None:
    """Config-driven build using PluginRegistry and ConfigPipelineBuilder."""
    orchestrator = PipelineOrchestrator()
    try:
        print_example_header("config builder")
        outputs = orchestrator.run(build_config_context(), pipeline_id="config-demo")
        print_result_summary(outputs)
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
            print(f"snapshot {snapshot.pipeline_id}: state={snapshot.state}, attempt={snapshot.attempt}, error={snapshot.error}")
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


def print_result_summary(outputs: PipelineOutputs) -> None:
    for result in outputs.results:
        if isinstance(result, TwoDimGraphData):
            print(f"- {result.label}: points={len(result.y)}, y={format_series(result.y)}, metadata={format_metadata(result.metadata)}")
        else:
            print(f"- {type(result).__name__}")
    print(f"- final artifacts: {len(outputs.final_artifacts)}")
    print(f"- debug artifacts: {len(outputs.debug_artifacts)}")


def print_event(event: Event) -> None:
    pipeline_id = event.payload.get("pipeline_id", "-")
    print(f"event {event.event_type} from {event.source} pipeline={pipeline_id}")


def format_series(values: list[float], max_items: int = 8) -> list[float | str]:
    if len(values) <= max_items:
        return values
    head = [round(value, 3) for value in values[: max_items // 2]]
    tail = [round(value, 3) for value in values[-max_items // 2 :]]
    return [*head, "...", *tail]


def format_metadata(metadata: dict) -> dict:
    return {key: round(float(value), 3) if isinstance(value, np.generic) else value for key, value in metadata.items()}


def wait_until_idle(runner: ThreadedPipelineRunner, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while runner.active_ids() and time.monotonic() < deadline:
        time.sleep(0.01)


def _can_show_preview() -> bool:
    """
    Probe whether OpenCV HighGUI is usable in this session.

    The check is performed in a subprocess so a failing GUI backend cannot
    crash the current Python process. When the probe fails, the demo falls
    back to headless execution and still returns tracking results.
    """
    probe = (
        "import cv2, numpy as np; "
        "img = np.zeros((1, 1, 3), dtype=np.uint8); "
        "cv2.imshow('sef-preview-probe', img); "
        "cv2.waitKey(1); "
        "cv2.destroyAllWindows()"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return False
    return completed.returncode == 0


def full_simple_example() -> None:
    """Run a compact tour of the core usage modes."""
    example_sync_run()
    example_fluent_builder()
    example_config_builder()
    example_async_multi_pipeline()
    example_event_driven_trigger_and_lifecycle()
    example_visualizer_targeting()


def single_example() -> None:
    example_sync_run_2()


if __name__ == "__main__":
    single_example()
