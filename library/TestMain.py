from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from library.analyzers.MultipleDistanceAnalyzer import MultipleDistanceAnalyzer
from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.utils.OpenCVMultiStartBoxSelector import OpenCVMultiStartBoxSelector
from library.core.utils.OpenCVStartBoxSelector import OpenCVStartBoxSelector
from library.frame_cleaners.OpenCVBackgroundSubtractionFrameCleaner import OpenCVBackgroundSubtractionFrameCleaner
from library.frame_cleaners.OpenCVHistogramEqualizationFrameCleaner import OpenCVHistogramEqualizationFrameCleaner
from library.frame_cleaners.OpenCVResizeFrameCleaner import OpenCVResizeFrameCleaner
from library.frame_cleaners.OpenCVZoomFrameCleaner import OpenCVZoomFrameCleaner
from library.frame_cleaners.SmoothingFrameCleaner import SmoothingFrameCleaner
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.live_analyzers.LiveVerticalPositionAnalyzer import LiveVerticalPositionAnalyzer
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.signal_extractors.OpenCVMultiManualSignalExtractor import OpenCVMultiManualSignalExtractor
from library.signal_extractors.OpenCVMultiObjectSignalExtractor import OpenCVMultiObjectSignalExtractor
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer


def build_fluent_context(video_path, zoom_box, start_box, start_boxes, resize) -> PipelineContext:
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                video_path,
                config={
                    "max_frames": 480  # (480 Frame)/(24 FPS) = 20 secondi
                },
            )
        )
        .add_frame_cleaner(OpenCVResizeFrameCleaner(resize))
        .add_frame_cleaner(OpenCVZoomFrameCleaner(zoom_box))
        .with_signal_extractor(OpenCVBufferedSignalExtractor(
            start_box=start_box,
            live_analyzer=LiveVerticalPositionAnalyzer(),
            config={
                "show": True,
                "show_graph": True,
            })
        )
        .with_analyzers([VerticalPositionAnalyzer()])
        .add_visualizer_for_results(MatplotlibFunctionVisualizer(), [0])
        .build_context()
    )


def main():
    #TODO SAM da problemi di import dopo che ho messo il live analyzer

    BASE_DIR = Path(__file__).resolve().parent
    video_path = BASE_DIR.parent / "videos" / ("Crowd.mp4")

    resize = (800, 600)

    zoom_box = None
    zoom_box = OpenCVStartBoxSelector().select_start(str(video_path), resize)

    start_box = None
    start_box = OpenCVStartBoxSelector().select_start(str(video_path), frame_cleaners=[OpenCVResizeFrameCleaner(resize), OpenCVZoomFrameCleaner(zoom_box)])
    #number_of_boxes = int(input("How many boxes would you like?: "))
    start_boxes = None
    #start_boxes = OpenCVMultiStartBoxSelector().select_start(str(video_path), number_of_boxes, frame_cleaners=[OpenCVResizeFrameCleaner(resize), OpenCVZoomFrameCleaner(zoom_box)])

    orchestrator = PipelineOrchestrator()
    try:
        outputs = orchestrator.run(
            build_fluent_context(video_path, zoom_box, start_box, start_boxes, resize),
            pipeline_id="2",
        )
    finally:
        orchestrator.shutdown()

    for image_artifact in outputs.artifacts:
        image = Image.open(io.BytesIO(image_artifact.data))
        image.show()


if __name__ == "__main__":
    main()
