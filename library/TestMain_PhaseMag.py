from __future__ import annotations

import logging
import sys
from pathlib import Path

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
from library.frame_processors.PhaseMagnificationFrameProcessor import PhaseMagnificationFrameProcessor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def build_phase_mag_pipeline(
    video_path: Path,
    output_path: Path,
) -> PipelineContext:

    return (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                video_path,
                config={"max_frames": 300},
            )
        )
        .add_frame_processor(
            PhaseMagnificationFrameProcessor(
                magnification_factor=15.0,
                low_cutoff_hz=0.4,
                high_cutoff_hz=3.0,
                fps=10.0,
            )
        )
        .add_frame_exporter(OpenCVFrameBufferVideoExporter(output_path, fps=10.0, max_exported_frames=300))
        .with_signal_extractor(NoSignalExtractor())
        .add_analyzer(NoAnalyzer())
        .build_context()
    )


def TestMain_PhaseMag() -> None:

    context = build_phase_mag_pipeline(
        video_path=PROJECT_ROOT / "videos" / "Tower.mp4",
        output_path=PROJECT_ROOT / "output" / "phase_mag_videos" / "Tower_PhaseMag.mp4",
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
