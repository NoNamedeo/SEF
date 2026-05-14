from __future__ import annotations

import io
from pathlib import Path

import cv2
from PIL import Image

from library.analyzers.VerticalPositionStreamAnalyzer import VerticalPositionStreamAnalyzer
from library.core.enum.FrameRotation import FrameRotation
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.FrameProcessingStage import FrameProcessingStage
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.core.utils.OpenCVMaskSelector import OpenCVMaskSelector
from library.core.utils.OpenCVStartBoxSelector import OpenCVStartBoxSelector
from library.core.visualization.VisualArtifact import ImageArtifact
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.frame_processors.OpenCVResizeFrameProcessor import OpenCVResizeFrameProcessor
from library.frame_processors.OpenCVRotateFrameProcessor import OpenCVRotateFrameProcessor
from library.signal_extractors.OpenCVStreamSignalExtractor import OpenCVStreamSignalExtractor
from library.visualizers.MatplotlibFunctionStreamVisualizer import MatplotlibFunctionStreamVisualizer


def _build_stream_context(
    video_path: Path,
    start_box: tuple[int, int, int, int],
    *,
    resize: tuple[int, int],
    rotation: FrameRotation,
) -> PipelineContext:
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                path=video_path,
                config={
                    "max_frames": 120,
                },
            )
        )
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVRotateFrameProcessor(rotation=rotation)))
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVResizeFrameProcessor(resize)))
        .with_signal_extractor(
            OpenCVStreamSignalExtractor(
                tracker_type="CSRT",
                start_box=start_box,
                config={
                    "show": False,
                    "show_graph": False,
                    "show_contours": False,
                },
            )
        )
        .with_analyzers([VerticalPositionStreamAnalyzer()])
        .add_visualizer_for_results(MatplotlibFunctionStreamVisualizer(), [0])
        .build_context()
    )


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


def _export_cleaned_video(context: PipelineContext, output_path: Path) -> Path:
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


def StreamTestMain() -> None:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    video_path = project_root / "videos" / "Tower.mp4"
    output_video_path = project_root / "output" / "visualization" / "Tower_stream_cleaned.mp4"

    resize = (1080, 1920)
    rotation = FrameRotation.ROTATE_90
    selection_processors = [
        OpenCVRotateFrameProcessor(rotation=rotation),
        OpenCVResizeFrameProcessor(resize),
    ]

    start_box = OpenCVStartBoxSelector().select_start(
        str(video_path),
        single_frame_processors=selection_processors,
    )

    stream_context = _build_stream_context(
        video_path,
        start_box,
        resize=resize,
        rotation=rotation,
    )

    orchestrator = PipelineOrchestrator()
    try:
        outputs = orchestrator.run(stream_context, pipeline_id="stream-tower")
        cleaned_video_path = _export_cleaned_video(stream_context, output_video_path)
    finally:
        orchestrator.shutdown()

    for artifact in [*outputs.final_artifacts, *outputs.debug_artifacts]:
        if isinstance(artifact, ImageArtifact):
            Image.open(io.BytesIO(artifact.data)).show()

    print(f"Cleaned stream video saved to: {cleaned_video_path}")


if __name__ == "__main__":
    StreamTestMain()
