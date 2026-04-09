from __future__ import annotations

from pathlib import Path

from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.core.pipeline.PipelineBuilder import PipelineBuilder
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.OpenCVMovingAverageCleaner import OpenCVMovingAverageCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer


def build_demo_pipeline(video_path: str, initial_box=(300, 200, 80, 120)):
    """Create a runnable pipeline using only the public core components."""
    return (
        PipelineBuilder()
            .with_frame_extractor(
                OpenCVBufferedFrameExtractor(
                    path=video_path,
                    config={"resize": (640, 480), "stride": 2, "max_frames": 420},
                )
            )
            .with_signal_extractor(
                OpenCVBufferedSignalExtractor(tracker_type="CSRT", start_box=initial_box)
            )
            .add_signal_cleaner(OpenCVMovingAverageCleaner(window_size=5))
            .add_analyzer(VerticalPositionAnalyzer(config={"use_timestamps": True}))
            .build()
    )


def main():
    root_dir = Path(__file__).resolve().parents[1]
    video_path = root_dir / "videos" / "Crowd.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    pipeline = build_demo_pipeline(str(video_path))
    analysis_results = pipeline.run()

    visualizer = MatplotlibFunctionVisualizer(config={"show": True})
    visualizer.visualize(analysis_results[0])


if __name__ == "__main__":
    main()
