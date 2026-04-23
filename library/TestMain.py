from __future__ import annotations

import io
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from library.analyzers.ArucoMarkerDisplacementAnalyzer import ArucoMarkerDisplacementAnalyzer
from library.analyzers.ArucoMarkerRelativeMotionAnalyzer import ArucoMarkerRelativeMotionAnalyzer
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.visualization.VisualArtifact import ImageArtifact, VideoArtifact
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_extractors.ArucoMarkerSignalExtractor import ArucoMarkerSignalExtractor
from library.visualizers.ArucoAnnotatedVideoVisualizer import ArucoAnnotatedVideoVisualizer
from library.visualizers.MatplotlibArucoMotionVisualizer import MatplotlibArucoMotionVisualizer


def build_fluent_context(video_path, zoom_box, start_box, start_boxes) -> PipelineContext:
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                video_path,
                config={
                    "max_frames": None  # (480 Frame)/(24 FPS) = 20 secondi
                },
            )
        )
        .with_signal_extractor(ArucoMarkerSignalExtractor(config={"show": True}))
        .with_analyzers([ArucoMarkerDisplacementAnalyzer(), ArucoMarkerRelativeMotionAnalyzer()])
        .add_visualizer_for_results((MatplotlibArucoMotionVisualizer()), [0, 1])
        .add_visualizer_for_results(ArucoAnnotatedVideoVisualizer(), [0])
        .build_context()
    )


def main():
    # TODO SAM da problemi di import dopo che ho messo il live analyzer

    BASE_DIR = Path(__file__).resolve().parent
    video_path = BASE_DIR.parent / "videos" / "ArUco_test_paper_2.MOV"

    resize = None
    start_box = None
    # number_of_boxes = int(input("How many boxes would you like?: "))
    start_boxes = None
    # start_boxes = OpenCVMultiStartBoxSelector().select_start(str(video_path), number_of_boxes, frame_cleaners=[OpenCVResizeFrameCleaner(resize), OpenCVZoomFrameCleaner(zoom_box)])  # noqa: E501

    orchestrator = PipelineOrchestrator()
    try:
        outputs = orchestrator.run(
            build_fluent_context(video_path, start_box, start_boxes, resize),
            pipeline_id="2",
        )
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


if __name__ == "__main__":
    main()
