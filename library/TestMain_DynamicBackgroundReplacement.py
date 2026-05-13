from __future__ import annotations

import io
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from library.analyzers.ArucoMarkerDisplacementAnalyzer import ArucoMarkerDisplacementAnalyzer
from library.analyzers.ArucoMarkerRelativeMotionAnalyzer import ArucoMarkerRelativeMotionAnalyzer
from library.analyzers.HoriziontalPositionAnalyzer import HorizontalPositionAnalyzer
from library.analyzers.TrackingPlaybackAnalyzer import TrackingPlaybackAnalyzer
from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.enum.FrameRotation import FrameRotation
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.core.utils.OpenCVMaskSelector import OpenCVMaskSelector
from library.core.utils.OpenCVStartBoxSelector import OpenCVStartBoxSelector
from library.core.visualization.VisualArtifact import ImageArtifact, VideoArtifact
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.frame_processors.OpenCVBackgroundReplacementFrameProcessor import OpenCVBackgroundReplacementFrameProcessor
from library.frame_processors.OpenCVDynamicBackgroundReplacementFrameProcessor import \
    OpenCVDynamicBackgroundReplacementFrameProcessor
from library.frame_processors.OpenCVInpaintFrameProcessor import OpenCVInpaintFrameProcessor
from library.frame_processors.OpenCVResizeFrameProcessor import OpenCVResizeFrameProcessor
from library.frame_processors.OpenCVRotateFrameProcessor import OpenCVRotateFrameProcessor
from library.live_analyzers.LiveVerticalPositionAnalyzer import LiveVerticalPositionAnalyzer
from library.signal_extractors.ArucoMarkerSignalExtractor import ArucoMarkerSignalExtractor
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.visualizers.ArucoAnnotatedVideoVisualizer import ArucoAnnotatedVideoVisualizer
from library.visualizers.MatplotlibArucoMotionVisualizer import MatplotlibArucoMotionVisualizer
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer
from library.visualizers.TrackingVideoVisualizer import TrackingVideoVisualizer


def build_fluent_context(video_path, zoom_box, start_box, start_boxes, resize, mask, BGimage) -> PipelineContext:
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                video_path,
                config={
                    "max_frames": 190  # (180 Frame)/(24 FPS) = 5 secondi
                },
            )
        )
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVRotateFrameProcessor(rotation=FrameRotation.ROTATE_90)))
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVResizeFrameProcessor(resize)))
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVDynamicBackgroundReplacementFrameProcessor(BGimage, mask, resize)))

        .with_signal_extractor(
                OpenCVBufferedSignalExtractor(
                start_box=start_box,
                live_analyzer=None,
                config={
                    "show": True
                },
            )
        )
        .with_analyzers([VerticalPositionAnalyzer()])
        .add_visualizer_for_results(MatplotlibFunctionVisualizer(), [0])
        .build_context()
    )


def main():
    BASE_DIR = Path(__file__).resolve().parent
    video_path = BASE_DIR.parent / "videos" / "Tower.mp4"

    background_image_path =  BASE_DIR.parent / "images" / "Tower_without_people.png"

    resize = (800, 600)
    rotation = FrameRotation.ROTATE_90

    zoom_box = None
    # zoom_box = OpenCVStartBoxSelector().select_start(
    #     str(video_path), single_frame_processors=[OpenCVRotateFrameProcessor(rotation), OpenCVResizeFrameProcessor(resize)]
    # )

    mask = None
    mask = OpenCVMaskSelector().select_mask(str(video_path), single_frame_processors=[OpenCVRotateFrameProcessor(rotation), OpenCVResizeFrameProcessor(resize)])# OpenCVZoomFrameCleaner(zoom_box)])

    start_box = None
    start_box = OpenCVStartBoxSelector().select_start(
        str(video_path), single_frame_processors=[OpenCVRotateFrameProcessor(rotation), OpenCVResizeFrameProcessor(resize)] # OpenCVZoomFrameCleaner(zoom_box)]
    )
    # number_of_boxes = int(input("How many boxes would you like?: "))
    start_boxes = None
    # start_boxes = OpenCVMultiStartBoxSelector().select_start(str(video_path), number_of_boxes, frame_cleaners=[OpenCVResizeFrameCleaner(resize), OpenCVZoomFrameCleaner(zoom_box)])  # noqa: E501

    orchestrator = PipelineOrchestrator()
    try:
        outputs = orchestrator.run(
            build_fluent_context(video_path, zoom_box, start_box, start_boxes, resize, mask, background_image_path),
            pipeline_id="2",
        )
    finally:
        orchestrator.shutdown()

    for artifact in [*outputs.final_artifacts, *outputs.debug_artifacts]:
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
