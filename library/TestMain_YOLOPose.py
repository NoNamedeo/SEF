from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from library.visualizers.OpenCVCOCOTennisPoseRealtimeVisualizer import OpenCVCOCOTennisPoseRealtimeVisualizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# Script entrypoint: package imports need the project root when run as `python library/...`.
from library.analyzers.COCOPoseStreamAnalyzer import COCOPoseStreamAnalyzer  # noqa: E402
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder  # noqa: E402
from library.core.pipeline.PipelineContext import PipelineContext  # noqa: E402
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator  # noqa: E402
from library.frame_extractors.OpenCVWebcamFrameExtractor import OpenCVWebcamFrameExtractor  # noqa: E402
from library.signal_extractors.YOLOSkeletonCOCOStreamSignalExtractor import (  # noqa: E402
    YOLOSkeletonCOCOStreamSignalExtractor,
)
from library.visualizers.OpenCVCOCOPoseRealtimeVisualizer import OpenCVCOCOPoseRealtimeVisualizer  # noqa: E402

DEFAULT_CAMERA_INDEX = 0
DEFAULT_MAX_FRAMES = 9999
DEFAULT_MODEL_NAME = "yolo11s-pose.pt"


def build_yolo_pose_webcam_pipeline(
    *,
    camera_index: int = DEFAULT_CAMERA_INDEX,
    max_frames: int = DEFAULT_MAX_FRAMES,
    model_name: str = DEFAULT_MODEL_NAME,
    mirror: bool = False,
    light: bool = False,
) -> PipelineContext:
    """Build the existing SEF streaming pipeline for a realtime YOLO pose smoke test."""

    return (
        FluentPipelineBuilder()
        .with_stream_runtime(
            {
                "frame_buffer_size": 1,
                "signal_buffer_size": 1,
                "latency_policy": {"name": "drop_oldest"},
            }
        )
        .with_frame_extractor(
            OpenCVWebcamFrameExtractor(
                camera_index=camera_index,
                config={
                    "max_frames": max_frames,
                    "mirror": mirror,
                },
            )
        )
        .with_signal_extractor(
            YOLOSkeletonCOCOStreamSignalExtractor(
                model_name=model_name,
                config={
                    "include_frame_image": not light,
                    "show_graph": False,
                },
            )
        )
        .add_analyzer(COCOPoseStreamAnalyzer(config={"retain_frames": False}))
        .add_visualizer_for_results(
            OpenCVCOCOTennisPoseRealtimeVisualizer(
                config={
                    "draw_source_frame": not light,
                },
            ),
            [0],
        )
        .build_context()
    )


def TestMain_YOLOPose() -> None:
    """Run the webcam -> YOLO pose -> realtime keypoint visualization pipeline."""

    args = _parse_args()
    context = build_yolo_pose_webcam_pipeline(
        camera_index=args.camera_index,
        max_frames=args.max_frames,
        model_name=args.model,
        mirror=args.mirror,
        light=args.light,
    )

    orchestrator = PipelineOrchestrator()
    try:
        outputs = orchestrator.run(context, pipeline_id="yolo-pose-webcam")

        print("Results summary:  \n")
        print("Latency policy metrics: ", outputs.metadata.execution_metadata["latency_policy_metrics"], ".\n")
    finally:
        orchestrator.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing SEF realtime webcam pipeline with YOLO pose keypoint visualization.",
    )
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--max-frames", type=_positive_int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument(
        "--light",
        action="store_true",
        help="Render only the pose layer on a synthetic canvas without carrying source frames.",
    )
    return parser.parse_args()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0.")
    return parsed


if __name__ == "__main__":
    TestMain_YOLOPose()
