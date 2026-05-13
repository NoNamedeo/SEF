from __future__ import annotations

import io
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from PIL import Image

from library.analyzers.ArucoMarkerDisplacementAnalyzer import ArucoMarkerDisplacementAnalyzer
from library.analyzers.ArucoMarkerRelativeMotionAnalyzer import ArucoMarkerRelativeMotionAnalyzer
from library.analyzers.HoriziontalPositionAnalyzer import HorizontalPositionAnalyzer
from library.analyzers.NoAnalyzer import NoAnalyzer
from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.enum.FrameRotation import FrameRotation
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.FrameProcessingStage import FrameProcessingStage
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.core.utils.OpenCVMaskSelector import OpenCVMaskSelector
from library.core.utils.OpenCVStartBoxSelector import OpenCVStartBoxSelector
from library.core.visualization.VisualArtifact import ImageArtifact, VideoArtifact
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.frame_processors.OpenCVBackgroundReplacementFrameProcessor import OpenCVBackgroundReplacementFrameProcessor
from library.frame_processors.OpenCVInpaintFrameProcessor import OpenCVInpaintFrameProcessor
from library.frame_processors.OpenCVResizeFrameProcessor import OpenCVResizeFrameProcessor
from library.frame_processors.OpenCVRotateFrameProcessor import OpenCVRotateFrameProcessor
from library.live_analyzers.LiveVerticalPositionAnalyzer import LiveVerticalPositionAnalyzer
from library.signal_extractors.ArucoMarkerSignalExtractor import ArucoMarkerSignalExtractor
from library.signal_extractors.NoSignalExtractor import NoSignalExtractor
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.visualizers.ArucoAnnotatedVideoVisualizer import ArucoAnnotatedVideoVisualizer
from library.visualizers.MatplotlibArucoMotionVisualizer import MatplotlibArucoMotionVisualizer
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

def _resolve_output_fps(context: PipelineContext, fallback_fps: float = 24.0) -> float:
    frame_extractor = context.frame_extractor
    source_path = getattr(frame_extractor, "path", None)
    stride = int(getattr(frame_extractor, "stride", 1) or 1)

    if not source_path:
        return fallback_fps

    cap = cv2.VideoCapture(str(source_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    if fps <= 0:
        return fallback_fps
    return max(fps / stride, 1.0)


def export_cleaned_video(context: PipelineContext, output_path: Path) -> Path:
    buffer = context.frame_extractor.extract()
    processed_buffer = FrameProcessingStage().apply(buffer, context.frame_processors)
    processed_frames = list(processed_buffer)

    if not processed_frames:
        raise ValueError("No processed frames available to export.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = processed_frames[0].image
    height, width = first_frame.shape[:2]
    fps = _resolve_output_fps(context)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")

    try:
        for frame in processed_frames:
            writer.write(frame.image)
    finally:
        writer.release()

    return output_path

def build_fluent_context(video_path, zoom_box, start_box, start_boxes, resize, mask, BGimage) -> PipelineContext:
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                video_path,
                config={
                    "max_frames": (180*24 + 150)  # (180 Frame)/(24 FPS) = 5 secondi
                },
            )
        )
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVRotateFrameProcessor(rotation=FrameRotation.ROTATE_90)))
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVResizeFrameProcessor(resize)))
        #.add_frame_processor(SingleFrameProcessorAdapter(OpenCVInpaintFrameProcessor(mask, radius=3, method=1)))
        #.add_frame_processor(SingleFrameProcessorAdapter(ColorStabilizationFrameCleaner()))
        #.add_frame_processor(SingleFrameProcessorAdapter(OpenCVZoomFrameCleaner(zoom_box)))
        #.add_frame_processor(SingleFrameProcessorAdapter(OpenCVDynamicBackgroundReplacementFrameCleaner(BGimage, mask, resize)))
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVBackgroundReplacementFrameProcessor(BGimage, mask, resize)))
        #.add_frame_processor(SingleFrameProcessorAdapter(OpenCVHistogramEqualizationFrameCleaner()))
        #.add_frame_processor(SingleFrameProcessorAdapter(OpenCVBackgroundSubtractionFrameCleaner()))

        .with_signal_extractor(
            NoSignalExtractor()
            # OpenCVBufferedSignalExtractor(
            #     start_box=start_box,
            #     live_analyzer=None,
            #     config={
            #         "show": 9999999,
            #         "show_graph": False
            #     },
            # )
        )
        #.with_analyzers([VerticalPositionAnalyzer()])
        .with_analyzers([NoAnalyzer()])
        #.add_visualizer_for_results(MatplotlibFunctionVisualizer(), [0])
        #.add_visualizer_for_results(MatplotlibArucoMotionVisualizer(), [0])
        .build_context()
    )


def main():
    BASE_DIR = Path(__file__).resolve().parent
    video_path = BASE_DIR.parent / "videos" / "Tower.mp4"

    background_image_path =  BASE_DIR.parent / "images" / "Tower_without_people.png"

    resize = (1080, 1920)
    rotation = FrameRotation.ROTATE_90

    zoom_box = None
    zoom_box = OpenCVStartBoxSelector().select_start(
        str(video_path), single_frame_processors=[OpenCVRotateFrameProcessor(rotation), OpenCVResizeFrameProcessor(resize)]
    )

    mask = None
    mask = OpenCVMaskSelector().select_mask(str(video_path), single_frame_processors=[OpenCVRotateFrameProcessor(rotation), OpenCVResizeFrameProcessor(resize)])# OpenCVZoomFrameCleaner(zoom_box)])

    start_box = None
    start_box = OpenCVStartBoxSelector().select_start(
        str(video_path), single_frame_processors=[OpenCVRotateFrameProcessor(rotation), OpenCVResizeFrameProcessor(resize)] # OpenCVZoomFrameCleaner(zoom_box)]
    )
    # number_of_boxes = int(input("How many boxes would you like?: "))
    start_boxes = None
    # start_boxes = OpenCVMultiStartBoxSelector().select_start(str(video_path), number_of_boxes, frame_cleaners=[OpenCVResizeFrameCleaner(resize), OpenCVZoomFrameCleaner(zoom_box)])  # noqa: E501

    pipeline_context = build_fluent_context(video_path, zoom_box, start_box, start_boxes, resize, mask, background_image_path)
    cleaned_video_path = BASE_DIR.parent / "output" / "cleaned_videos" / f"{video_path.stem}_without_people.mp4"

    orchestrator = PipelineOrchestrator()
    try:
        outputs = orchestrator.run(
            pipeline_context,
            pipeline_id="2",
        )
        exported_video_path = export_cleaned_video(pipeline_context, cleaned_video_path)
    finally:
        orchestrator.shutdown()

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

    print(f"Cleaned video saved to: {exported_video_path}")


if __name__ == "__main__":
    main()
