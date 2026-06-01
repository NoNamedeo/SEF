from __future__ import annotations

import logging
import sys
from pathlib import Path

from library.core.enum.FrameRotation import FrameRotation
from library.core.pipeline.SingleFrameProcessorAdapter import SingleFrameProcessorAdapter
from library.core.utils.OpenCVMaskSelector import OpenCVMaskSelector
from library.frame_processors.ColorStabilizationFrameProcessor import ColorStabilizationFrameProcessor
from library.frame_processors.OpenCV.OpenCVBackgroundReplacementFrameProcessor import \
    OpenCVBackgroundReplacementFrameProcessor
from library.frame_processors.OpenCV.OpenCVResizeFrameProcessor import OpenCVResizeFrameProcessor
from library.frame_processors.OpenCV.OpenCVRotateFrameProcessor import OpenCVRotateFrameProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from library.analyzers.NoAnalyzer import NoAnalyzer
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.exporters.OpenCVFrameBufferVideoExporter import OpenCVFrameBufferVideoExporter
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_extractors.NoSignalExtractor import NoSignalExtractor
from library.frame_processors.motion_magnification.PhaseMagnificationFrameProcessor import PhaseMagnificationFrameProcessor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def build_phase_mag_pipeline(
    video_path: Path,
    output_path: Path,
    background_image_path: Path,
    mask,
    resize,
) -> PipelineContext:

    return (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                video_path,
                config={"max_frames": 300}, #300 per phase_mag sembra essere il massimo per il mio pc (alej), prima che collassi
            )
        )
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVRotateFrameProcessor(rotation=FrameRotation.ROTATE_90)))
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVResizeFrameProcessor(resize)))
        .add_frame_processor(SingleFrameProcessorAdapter(OpenCVBackgroundReplacementFrameProcessor(
            background_image_path=str(background_image_path),
            mask=mask,
            resize=resize
        )))
        .add_frame_processor(SingleFrameProcessorAdapter(ColorStabilizationFrameProcessor()))
        .add_frame_processor(
            PhaseMagnificationFrameProcessor(
                magnification_factor=15.0,
                low_cutoff_hz=0.4,
                high_cutoff_hz=3.0,
                fps=20.0,
            )
        )
        .add_frame_exporter(OpenCVFrameBufferVideoExporter(output_path, fps=20.0, max_exported_frames=300))
        .with_signal_extractor(NoSignalExtractor())
        .add_analyzer(NoAnalyzer())
        .build_context()
    )


def TestMain_PhaseMag() -> None:

    video_path = PROJECT_ROOT / "videos" / "Tower_3.mp4"
    output_path = PROJECT_ROOT / "output" / "phase_mag_videos" / "Tower_3_PhaseMag_Static.mp4"
    background_image_path = PROJECT_ROOT / "images" / "Tower_without_people_3.png"

    resize = (800, 600)

    mask = None
    mask = OpenCVMaskSelector().select_mask(
        str(video_path),
        single_frame_processors=[
            OpenCVRotateFrameProcessor(rotation=FrameRotation.ROTATE_90),
            OpenCVResizeFrameProcessor(resize),
        ])

    context = build_phase_mag_pipeline(
        video_path=video_path,
        output_path=output_path,
        background_image_path=background_image_path,
        mask=mask,
        resize=resize
    )

    orchestrator = PipelineOrchestrator()
    try:
        outputs = orchestrator.run(context, pipeline_id="phasemag")
        print("Results summary:  \n")
        print("Latency policy metrics: ", outputs.metadata.execution_metadata["latency_policy_metrics"], ".\n")
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    TestMain_PhaseMag()
