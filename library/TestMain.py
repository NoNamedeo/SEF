from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from PIL import Image

from library.analyzers.NoAnalyzer import NoAnalyzer
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.enum.FrameRotation import FrameRotation
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.IntermediateFrameCapture import IntermediateFrameCaptureConfig
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.core.utils.OpenCVMaskSelector import OpenCVMaskSelector
from library.core.visualization.PipelineOutputs import PipelineOutputs
from library.core.visualization.VisualArtifact import DeferredVideoArtifact, ImageArtifact, VideoArtifact, VideoFileArtifact
from library.exporters.OpenCVFrameBufferVideoExporter import OpenCVFrameBufferVideoExporter
from library.frame_processors.DynamicObjectRemovalFrameProcessor import DynamicObjectRemovalFrameProcessor
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.frame_processors.ColorStabilizationFrameProcessor import ColorStabilizationFrameProcessor
from library.frame_processors.OpenCVBackgroundReplacementFrameProcessor import OpenCVBackgroundReplacementFrameProcessor
from library.frame_processors.OpenCVGrayFrameProcessor import OpenCVGrayFrameProcessor
from library.frame_processors.OpenCVResizeFrameProcessor import OpenCVResizeFrameProcessor
from library.frame_processors.OpenCVRotateFrameProcessor import OpenCVRotateFrameProcessor
from library.frame_processors.OpenCVZoomFrameProcessor import OpenCVZoomFrameProcessor
from library.signal_extractors.NoSignalExtractor import NoSignalExtractor
from library.visualizers.IntermediateFramesGridVisualizer import IntermediateFramesGridVisualizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FALLBACK_FPS = 24.0
DEFAULT_MAX_FRAMES = 99
DEFAULT_DEBUG_MAX_STORED_FRAMES = 12
DEFAULT_STREAM_BUFFER_SIZE = 8
DEFAULT_RESIZE = (1080, 1920)
FAST_DYNAMIC_DEMO_RESIZE = (540, 960)
FAST_DYNAMIC_DEMO_MAX_FRAMES = 99
FAST_DYNAMIC_DEMO_STRIDE = 3
FAST_DYNAMIC_DEMO_MAX_SAMPLED_FRAMES = 24
PIPELINE_ID = "background-replacement-demo"
RUN_DYNAMIC_OBJECT_REMOVAL_DEMO = True
log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BackgroundReplacementPipelineConfig:
    """
    Runtime configuration for the Tower background-replacement pipeline.

    Paths and visual debug settings are kept outside the composition logic so
    the same pipeline can be reused from scripts, tests, or a future UI.
    """

    video_path: Path
    background_image_path: Path
    cleaned_video_path: Path
    dynamic_cleaned_video_path: Path
    artifact_output_dir: Path
    resize: tuple[int, int] = DEFAULT_RESIZE
    rotation: FrameRotation = FrameRotation.ROTATE_90
    max_frames: int = DEFAULT_MAX_FRAMES
    frame_stride: int = 1
    debug_sampling_interval: int = 45
    debug_max_stored_frames: int = DEFAULT_DEBUG_MAX_STORED_FRAMES
    stream_buffer_size: int = DEFAULT_STREAM_BUFFER_SIZE
    fallback_fps: float = DEFAULT_FALLBACK_FPS

    @classmethod
    def default(cls) -> BackgroundReplacementPipelineConfig:
        """Return the demo defaults used by the original script."""
        return cls(
            video_path=PROJECT_ROOT / "videos" / "Tower.mp4",
            background_image_path=PROJECT_ROOT / "images" / "Tower_without_people.png",
            cleaned_video_path=PROJECT_ROOT / "output" / "cleaned_videos" / "Tower_without_people.mp4",
            dynamic_cleaned_video_path=PROJECT_ROOT / "output" / "cleaned_videos" / "Tower_dynamic_object_removal.mp4",
            artifact_output_dir=PROJECT_ROOT / "output" / "visualizations" / "background_replacement",
            max_frames=9999,
        )

    def validate(self, *, require_background_image: bool = True) -> None:
        """Fail fast before opening OpenCV UI selectors or executing the pipeline."""
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {self.video_path}")
        if require_background_image and not self.background_image_path.is_file():
            raise FileNotFoundError(f"Background image not found: {self.background_image_path}")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be greater than 0.")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride must be greater than 0.")
        if self.debug_sampling_interval <= 0:
            raise ValueError("debug_sampling_interval must be greater than 0.")
        if self.debug_max_stored_frames < 0:
            raise ValueError("debug_max_stored_frames cannot be negative.")
        if self.stream_buffer_size <= 0:
            raise ValueError("stream_buffer_size must be greater than 0.")


class BackgroundReplacementPipelineFactory:
    """Compose the background-replacement pipeline from explicit dependencies."""

    def __init__(self, config: BackgroundReplacementPipelineConfig) -> None:
        self._config = config

    def build(self, mask) -> PipelineContext:
        """Create a validated PipelineContext for one selected replacement mask."""
        return (
            FluentPipelineBuilder()
            .with_frame_extractor(
                OpenCVBufferedFrameExtractor(
                    self._config.video_path,
                    buffer=FrameBuffer(buffer_size=self._config.stream_buffer_size),
                    config={
                        "max_frames": self._config.max_frames,
                        "stride": self._config.frame_stride,
                    },
                )
            )
            .with_frame_processors(
                [
                    SingleFrameProcessorAdapter(OpenCVRotateFrameProcessor(rotation=self._config.rotation)),
                    SingleFrameProcessorAdapter(OpenCVResizeFrameProcessor(self._config.resize)),
                    SingleFrameProcessorAdapter(
                        OpenCVBackgroundReplacementFrameProcessor(
                            str(self._config.background_image_path),
                            mask,
                            self._config.resize,
                        )
                    ),
                    SingleFrameProcessorAdapter(OpenCVGrayFrameProcessor()),
                ]
            )
            .add_frame_exporter(
                OpenCVFrameBufferVideoExporter(
                    output_path=self._config.cleaned_video_path,
                    fps=resolve_output_fps(
                        self._config.video_path,
                        stride=self._config.frame_stride,
                        fallback_fps=self._config.fallback_fps,
                    ),
                    title=f"{self._config.video_path.stem} without people",
                    description="Background-replacement final video.",
                    max_exported_frames=self._config.max_frames,
                )
            )
            .with_signal_extractor(NoSignalExtractor())
            .with_analyzers([NoAnalyzer()])
            .with_intermediate_frame_capture(self._intermediate_capture_config())
            .add_intermediate_frame_visualizer(self._intermediate_frame_visualizer())
            .build_context()
        )

    def build_dynamic_object_removal_demo(self) -> PipelineContext:
        """Create a quick demo pipeline for temporal dynamic-object removal."""
        max_frames = min(self._config.max_frames, FAST_DYNAMIC_DEMO_MAX_FRAMES)
        stride = max(self._config.frame_stride, FAST_DYNAMIC_DEMO_STRIDE)
        log.info(
            "Dynamic object removal demo: resize=%s max_frames=%s stride=%s max_sampled_frames=%s",
            FAST_DYNAMIC_DEMO_RESIZE,
            max_frames,
            stride,
            FAST_DYNAMIC_DEMO_MAX_SAMPLED_FRAMES,
        )
        return (
            FluentPipelineBuilder()
            .with_frame_extractor(
                OpenCVBufferedFrameExtractor(
                    self._config.video_path,
                    config={
                        "max_frames": max_frames,
                        "stride": stride,
                    },
                )
            )
            .with_frame_processors(
                [
                    SingleFrameProcessorAdapter(OpenCVRotateFrameProcessor(rotation=self._config.rotation)),
                    SingleFrameProcessorAdapter(OpenCVResizeFrameProcessor(FAST_DYNAMIC_DEMO_RESIZE)),
                    SingleFrameProcessorAdapter(
                        ColorStabilizationFrameProcessor(
                            color_space="LAB",
                            techniques=[
                                "luminance_normalization",
                                "temporal_smoothing",
                                "gamma_correction",
                            ],
                            stabilization_strength=0.45,
                            temporal_alpha=0.08,
                            luminance_max_shift=18.0,
                            gamma_limits=(0.85, 1.20),
                            stabilize_chroma=True,
                            chroma_strength=0.15,
                            max_chroma_shift=5.0,
                            scene_change_luminance_threshold=35.0,
                            emit_metrics=True,
                        )
                    ),
                    DynamicObjectRemovalFrameProcessor(
                        sampling_stride=1,
                        max_sampled_frames=200,
                        difference_threshold=0.5,
                        difference_mode="luma",
                        morph_kernel_size=5,
                        opening_iterations=1,
                        closing_iterations=3,
                        dilation_iterations=2,
                        max_sequence_bytes=4096 * 1024 * 1024,
                        min_component_area=50,
                        max_processed_frames=9999,
                        emit_intermediate_artifacts=False,
                    ),
                    SingleFrameProcessorAdapter(
                        ColorStabilizationFrameProcessor(
                            color_space="LAB",
                            techniques=[
                                "temporal_smoothing",
                                "gamma_correction",
                            ],
                            stabilization_strength=0.25,
                            temporal_alpha=0.12,
                            luminance_max_shift=10.0,
                            gamma_limits=(0.9, 1.12),
                            stabilize_chroma=True,
                            chroma_strength=0.10,
                            max_chroma_shift=3.0,
                            emit_metrics=False,
                        )
                    ),
                ]
            )
            .add_frame_exporter(
                OpenCVFrameBufferVideoExporter(
                    output_path=self._config.dynamic_cleaned_video_path,
                    fps=resolve_output_fps(
                        self._config.video_path,
                        stride=stride,
                        fallback_fps=self._config.fallback_fps,
                    ),
                    title=f"{self._config.video_path.stem} dynamic object removal",
                    description="Temporal median dynamic-object-removal final video.",
                    max_exported_frames=max_frames,
                )
            )
            .with_signal_extractor(NoSignalExtractor())
            .with_analyzers([NoAnalyzer()])
            .with_intermediate_frame_capture(self._intermediate_capture_config())
            .add_intermediate_frame_visualizer(self._intermediate_frame_visualizer())
            .build_context()
        )

    def _intermediate_capture_config(self) -> IntermediateFrameCaptureConfig:
        return IntermediateFrameCaptureConfig(
            enabled=True,
            sampling_interval=self._config.debug_sampling_interval,
            max_stored_frames=self._config.debug_max_stored_frames,
            include_original=True,
            export_directory=self._config.artifact_output_dir,
            lazy_saving=True,
        )

    @staticmethod
    def _intermediate_frame_visualizer() -> IntermediateFramesGridVisualizer:
        return IntermediateFramesGridVisualizer(
            config={
                "columns": 2,
                "max_artifacts": 12,
                "max_panel_width": 420,
                "max_cell_width": 700,
                "show_cell_labels": True,
            }
        )


class PipelineArtifactWriter:
    """Persist pipeline visual artifacts without opening platform-specific GUIs."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def write(self, outputs: PipelineOutputs) -> tuple[Path, ...]:
        """Save image and video artifacts and return their filesystem paths."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        artifacts = [*outputs.final_artifacts, *outputs.debug_artifacts]
        for index, artifact in enumerate(artifacts, start=1):
            if isinstance(artifact, ImageArtifact):
                path = self._output_dir / f"{index:02d}_{self._safe_stem(artifact.title or 'artifact')}.png"
                Image.open(io.BytesIO(artifact.data)).save(path)
                saved_paths.append(path)
            elif isinstance(artifact, VideoArtifact):
                path = self._output_dir / f"{index:02d}_{self._safe_stem(artifact.title or 'artifact')}.mp4"
                path.write_bytes(artifact.data)
                saved_paths.append(path)
            elif isinstance(artifact, VideoFileArtifact):
                saved_paths.append(Path(artifact.path))
            elif isinstance(artifact, DeferredVideoArtifact):
                saved_paths.append(artifact.materialize(self._output_dir))

        return tuple(saved_paths)

    @staticmethod
    def _safe_stem(value: str) -> str:
        return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value).strip("_") or "artifact"


def select_replacement_mask(config: BackgroundReplacementPipelineConfig):
    """Use the existing OpenCV selector on the same frame geometry as the pipeline."""
    selector_processors = [
        OpenCVRotateFrameProcessor(config.rotation),
        OpenCVResizeFrameProcessor(config.resize),
    ]
    return OpenCVMaskSelector().select_mask(
        str(config.video_path),
        single_frame_processors=selector_processors,
    )


def resolve_output_fps(video_path: Path, *, stride: int, fallback_fps: float) -> float:
    """Return the FPS for the exported processed stream."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()

    if fps <= 0:
        return fallback_fps
    return max(fps / stride, 1.0)


def run_pipeline(context: PipelineContext) -> PipelineOutputs:
    """Execute the context through the application-facing orchestrator."""
    orchestrator = PipelineOrchestrator()
    try:
        return orchestrator.run(context, pipeline_id=PIPELINE_ID)
    finally:
        orchestrator.shutdown()


def print_run_summary(cleaned_video_path: Path, saved_artifacts: Iterable[Path]) -> None:
    """Write a concise CLI summary for manual demo runs."""
    print(f"Cleaned video saved to: {cleaned_video_path}")
    for path in saved_artifacts:
        print(f"Visualization artifact saved to: {path}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    config = BackgroundReplacementPipelineConfig.default()
    config.validate(require_background_image=not RUN_DYNAMIC_OBJECT_REMOVAL_DEMO)

    factory = BackgroundReplacementPipelineFactory(config)
    if RUN_DYNAMIC_OBJECT_REMOVAL_DEMO:
        context = factory.build_dynamic_object_removal_demo()
        cleaned_video_path = config.dynamic_cleaned_video_path
    else:
        mask = select_replacement_mask(config)
        context = factory.build(mask)
        cleaned_video_path = config.cleaned_video_path
    outputs = run_pipeline(context)
    saved_artifacts = PipelineArtifactWriter(config.artifact_output_dir).write(outputs)

    print_run_summary(cleaned_video_path, saved_artifacts)


if __name__ == "__main__":
    main()
