from __future__ import annotations

import io
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.enum.FrameRotation import FrameRotation
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.core.utils.OpenCVStartBoxSelector import OpenCVStartBoxSelector
from library.core.visualization.VisualArtifact import ImageArtifact, VideoArtifact
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.frame_processors.ColorStabilizationFrameProcessor import ColorStabilizationFrameProcessor
from library.frame_processors.OpenCVResizeFrameProcessor import OpenCVResizeFrameProcessor
from library.frame_processors.OpenCVRotateFrameProcessor import OpenCVRotateFrameProcessor
from library.frame_processors.OpenCVZoomFrameProcessor import OpenCVZoomFrameProcessor
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.visualizers.IntermediateFramesGridVisualizer import IntermediateFramesGridVisualizer
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

COLOR_STABILIZATION_DEBUG_DIR = Path("outputs/color_stabilization_debug")


def build_fluent_context(video_path, zoom_box, start_box, start_boxes, resize) -> PipelineContext:
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                video_path,
                config={
                    "max_frames": 120  # (180 Frame)/(24 FPS) = 5 secondi
                },
            )
        )
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVRotateFrameProcessor(rotation=FrameRotation.ROTATE_90)))
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVResizeFrameProcessor(resize)))
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVZoomFrameProcessor(zoom_box)))
        .add_frame_processor(
            SingleFrameProcessorAdapter(
                ColorStabilizationFrameProcessor(
                    color_space="LAB",
                    techniques=(
                        "temporal_smoothing",
                        "luminance_normalization",
                        "histogram_normalization",
                        "gamma_correction",
                        "clahe",
                    ),
                    stabilization_strength=0.85,
                    temporal_alpha=0.92,
                    stabilize_chroma=True,
                    chroma_strength=0.20,
                    emit_metrics=True,
                    emit_comparison_overlay=True,
                    emit_intermediate_artifacts=True,
                )
            )
        )
        .with_intermediate_frame_capture(
            {
                "enabled": True,
                "sampling_interval": 5,
                "max_stored_frames": 12,
                "export_directory": str(COLOR_STABILIZATION_DEBUG_DIR / "frames"),
                "lazy_saving": True,
                "include_original": True,
                "metadata": {"debug_target": "ColorStabilizationFrameProcessor"},
            }
        )
        .add_intermediate_frame_visualizer(
            IntermediateFramesGridVisualizer(
                config={
                    "columns": 2,
                    "max_artifacts": 12,
                    "show_labels": True,
                    "show_cell_labels": True,
                    "include_overlays": True,
                    "max_panel_width": 480,
                    "max_cell_width": 900,
                }
            )
        )
        .with_signal_extractor(
            OpenCVBufferedSignalExtractor(
                start_box=start_box,
                config={
                    "show": True,
                    "show_graph": False,
                },
            )
        )
        .with_analyzers([VerticalPositionAnalyzer()])
        .add_visualizer_for_results(MatplotlibFunctionVisualizer(), [0])
        .build_context()
    )


def main():
    # TODO SAM da problemi di import dopo che ho messo il live analyzer

    BASE_DIR = Path(__file__).resolve().parent
    video_path = BASE_DIR.parent / "videos" / "sunset.mp4"

    resize = (800, 600)
    rotation = FrameRotation.ROTATE_90

    zoom_box = None
    zoom_box = OpenCVStartBoxSelector().select_start(
        str(video_path), single_frame_processors=[OpenCVRotateFrameProcessor(rotation), OpenCVResizeFrameProcessor(resize)]
    )

    start_box = None
    start_box = OpenCVStartBoxSelector().select_start(
        str(video_path),
        single_frame_processors=[OpenCVRotateFrameProcessor(rotation), OpenCVResizeFrameProcessor(resize), OpenCVZoomFrameProcessor(zoom_box)],
    )
    # number_of_boxes = int(input("How many boxes would you like?: "))
    start_boxes = None
    # start_boxes = OpenCVMultiStartBoxSelector().select_start(str(video_path), number_of_boxes, single_frame_processors=[OpenCVResizeFrameProcessor(resize), OpenCVZoomFrameProcessor(zoom_box)])  # noqa: E501

    orchestrator = PipelineOrchestrator()
    try:
        outputs = orchestrator.run(
            build_fluent_context(video_path, zoom_box, start_box, start_boxes, resize),
            pipeline_id="2",
        )
    finally:
        orchestrator.shutdown()

    exported_intermediate_frames = outputs.intermediate_frames.export()
    print(f"Intermediate color-stabilization frames exported: {len(exported_intermediate_frames)}")

    for artifact in outputs.artifacts:
        if isinstance(artifact, ImageArtifact):
            image = Image.open(io.BytesIO(artifact.data))

            image.show()

        elif isinstance(artifact, VideoArtifact):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(artifact.data)

                temp_path = tmp.name

            # apertura cross-platform

            if sys.platform == "darwin":  # Mac
                subprocess.run(["open", temp_path])

            elif sys.platform == "win32":
                subprocess.run(["start", temp_path], shell=True)

            else:  # Linux
                subprocess.run(["xdg-open", temp_path])


if __name__ == "__main__":
    main()
