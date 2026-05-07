from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.IntermediateFrameComposition import compose_side_by_side
from library.core.artifacts.MaskArtifacts import (
    FrameMaskArtifact,
    IntermediateFrameArtifact,
    IntermediateFrameOverlay,
)
from library.core.artifacts.Signal import Signal
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameCaptureContext
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.Pipeline import Pipeline
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry
from library.core.visualization.VisualArtifact import ImageArtifact
from library.visualizers.IntermediateFramesGridVisualizer import IntermediateFramesGridVisualizer
from library.visualizers.IntermediateFramesVisualizer import IntermediateFramesVisualizer


class DebugResult(IData):
    """Minimal analysis result used to keep pipeline tests focused."""

    def __init__(self, title: str = "debug-result") -> None:
        self.title = title
        self.metadata: dict[str, Any] = {}


class StaticFrameExtractor(IFrameExtractor):
    """Frame extractor that emits a deterministic in-memory sequence."""

    def __init__(self, frame_count: int, shape: tuple[int, int, int] = (6, 6, 3)) -> None:
        super().__init__()
        self._frame_count = frame_count
        self._shape = shape

    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(self._frame_count)
        for index in range(self._frame_count):
            image = np.full(self._shape, index, dtype=np.uint8)
            buffer.put(Frame(image=image, index=index, timestamp_seconds=index / 10.0))
        buffer.close()
        return buffer


class PassthroughSignalExtractor(ISignalExtractor):
    """Signal extractor that consumes the cleaned buffer and records frame means."""

    def __init__(self) -> None:
        super().__init__()
        self.frame_means: list[float] = []

    def extract(self, buffer: FrameBuffer) -> ISignal:
        for frame in buffer:
            self.frame_means.append(float(np.mean(frame.image)))
        return Signal([])


class ConstantAnalyzer(IAnalyzer):
    """Analyzer stub returning a single stable result."""

    def analyze(self, signal: ISignal) -> IData:
        return DebugResult()


class AddValueCleaner(IFrameCleaner):
    """Cleaner that adds a constant value to every pixel."""

    def __init__(self, value: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._value = value

    def clean(self, frame: Frame) -> Frame:
        return Frame(
            image=np.clip(frame.image + self._value, 0, 255).astype(np.uint8),
            index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            metadata={**frame.metadata, "added_value": self._value},
        )


class MaskEmittingCleaner(AddValueCleaner):
    """Cleaner that emits custom masks and overlays through the optional protocol."""

    def emit_intermediate_artifacts(
        self,
        original_frame: Frame,
        cleaned_frame: Frame,
        context: IntermediateFrameCaptureContext,
    ):
        mask = np.zeros(cleaned_frame.image.shape[:2], dtype=np.bool_)
        mask[:2, :2] = True
        overlay = np.zeros_like(cleaned_frame.image)
        overlay[:, :, 1] = 255
        return (
            IntermediateFrameArtifact(
                image=cleaned_frame.image,
                stage_name=context.stage_name,
                frame_index=context.frame_index,
                timestamp_seconds=context.timestamp_seconds,
                original_image=original_frame.image,
                masks=(FrameMaskArtifact(mask=mask, label="top-left"),),
                overlays=(IntermediateFrameOverlay(image=overlay, label="green-overlay", alpha=0.25),),
                stage_metadata={"custom_emitter": True},
                metadata={"cleaner_name": context.cleaner_name},
                config=context.cleaner_config,
            ),
        )


def test_pipeline_emits_intermediate_frame_artifacts_with_original_and_custom_debug_layers() -> None:
    signal_extractor = PassthroughSignalExtractor()
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StaticFrameExtractor(frame_count=3))
        .with_frame_cleaners([MaskEmittingCleaner(10, config={"debug": True})])
        .with_signal_extractor(signal_extractor)
        .add_analyzer(ConstantAnalyzer())
        .with_intermediate_frame_capture(
            {
                "enabled": True,
                "sampling_interval": 1,
                "max_stored_frames": 10,
                "include_original": True,
            }
        )
        .build_context()
    )

    outputs = Pipeline(context).run()

    assert len(outputs.results) == 1
    assert outputs.intermediate_frames.count == 3
    first = outputs.intermediate_frames.artifacts[0]
    assert first.stage_name.endswith("MaskEmittingCleaner")
    assert first.original_frame is not None
    assert first.original_frame.flags.writeable is False
    assert first.cleaned_frame[0, 0, 0] == 10
    assert len(first.masks) == 1
    assert len(first.overlays) == 1
    assert first.stage_metadata["custom_emitter"] is True
    assert signal_extractor.frame_means == [10.0, 11.0, 12.0]


def test_intermediate_frame_visualizers_render_multiple_cleaner_stages() -> None:
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StaticFrameExtractor(frame_count=2))
        .with_frame_cleaners([AddValueCleaner(2), AddValueCleaner(3)])
        .with_signal_extractor(PassthroughSignalExtractor())
        .add_analyzer(ConstantAnalyzer())
        .with_intermediate_frame_capture({"enabled": True, "sampling_interval": 1, "max_stored_frames": 10})
        .add_intermediate_frame_visualizer(
            IntermediateFramesGridVisualizer(
                config={
                    "columns": 2,
                    "max_artifacts": 4,
                    "max_panel_width": 80,
                    "max_cell_width": 160,
                }
            )
        )
        .build_context()
    )

    outputs = Pipeline(context).run()

    assert outputs.intermediate_frames.count == 4
    assert len(outputs.intermediate_frames.stage_names) == 2
    assert len(outputs.artifacts) == 1
    assert isinstance(outputs.artifacts[0], ImageArtifact)
    assert len(outputs.artifacts[0].data) > 0

    per_snapshot_artifacts = IntermediateFramesVisualizer(config={"max_artifacts": 2}).render(outputs.intermediate_frames)
    assert len(per_snapshot_artifacts) == 2
    assert all(isinstance(artifact, ImageArtifact) for artifact in per_snapshot_artifacts)


def test_intermediate_frame_capture_respects_sampling_and_max_stored_frames(tmp_path: Path) -> None:
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(StaticFrameExtractor(frame_count=6))
        .with_frame_cleaners([AddValueCleaner(1), AddValueCleaner(2)])
        .with_signal_extractor(PassthroughSignalExtractor())
        .add_analyzer(ConstantAnalyzer())
        .with_intermediate_frame_capture(
            {
                "enabled": True,
                "sampling_interval": 2,
                "max_stored_frames": 2,
                "export_directory": str(tmp_path),
                "lazy_saving": True,
            }
        )
        .build_context()
    )

    outputs = Pipeline(context).run()

    source_indexes = {
        artifact.stage_metadata["source_sequence_index"]
        for artifact in outputs.intermediate_frames.artifacts
    }
    assert source_indexes == {2, 4}
    assert outputs.intermediate_frames.count == 4
    assert list(tmp_path.iterdir()) == []

    exported_paths = outputs.intermediate_frames.export()
    assert len(exported_paths) == 8
    assert all(path.exists() for path in exported_paths)


def test_config_builder_parses_intermediate_frame_capture_and_visualizers(tmp_path: Path) -> None:
    registry = PluginRegistry()
    registry.register(PluginCategory.FRAME_EXTRACTOR, "static", lambda: StaticFrameExtractor(frame_count=1))
    registry.register(PluginCategory.FRAME_CLEANER, "add", lambda value: AddValueCleaner(value))
    registry.register(PluginCategory.SIGNAL_EXTRACTOR, "passthrough", PassthroughSignalExtractor)
    registry.register(PluginCategory.ANALYZER, "constant", ConstantAnalyzer)
    registry.register(PluginCategory.VISUALIZER, "debug_grid", IntermediateFramesGridVisualizer)

    context = ConfigPipelineBuilder(registry).build_context(
        {
            "pipeline": {
                "frame_extractor": {"name": "static"},
                "frame_cleaners": [{"name": "add", "params": {"value": 5}}],
                "signal_extractor": {"name": "passthrough"},
                "analyzers": [{"name": "constant"}],
                "intermediate_frames": {
                    "enabled": True,
                    "sampling_interval": 1,
                    "max_stored_frames": 3,
                    "export_directory": str(tmp_path),
                    "lazy_saving": True,
                    "visualizers": [{"name": "debug_grid", "params": {"columns": 1}}],
                },
            }
        }
    )

    outputs = Pipeline(context).run()

    assert context.intermediate_frame_capture.enabled is True
    assert context.intermediate_frame_capture.max_stored_frames == 3
    assert len(context.intermediate_frame_visualizers) == 1
    assert outputs.intermediate_frames.count == 1
    assert len(outputs.artifacts) == 1


def test_side_by_side_composer_returns_displayable_bgr_image() -> None:
    left = np.zeros((4, 4, 3), dtype=np.uint8)
    right = np.full((4, 4), 255, dtype=np.uint8)

    composed = compose_side_by_side([left, right], labels=["left", "right"], max_panel_width=8)

    assert composed.ndim == 3
    assert composed.shape[2] == 3
    assert composed.dtype == np.uint8
