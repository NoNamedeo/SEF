from __future__ import annotations

from pathlib import Path

from library.analyzers.DenseOpticalFlowVectorFieldAnalyzer import DenseOpticalFlowVectorFieldAnalyzer
from library.analyzers.MultiObjectBarrierCountingAnalyzer import (
    MultiObjectBarrierCountingAnalyzer,
)
from library.analyzers.VerticalFrequencyAnalyzer import VerticalFrequencyAnalyzer
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.Pipeline import Pipeline
from library.core.utils.OpenCVBarrierSelector import OpenCVBarrierSelector
from library.core.utils.OpenCVStartBoxSelector import OpenCVStartBoxSelector
from library.frame_cleaners.OpenCVGrayFrameCleaner import OpenCVGrayFrameCleaner
from library.frame_cleaners.SmoothingFrameCleaner import SmoothingFrameCleaner
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.MovingAverageCleaner import MovingAverageCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.signal_extractors.OpenCVDenseOpticalFlowSignalExtractor import OpenCVDenseFarnebackSignalExtractor
from library.signal_extractors.OpenCVMultiObjectSignalExtractor import OpenCVMultiObjectSignalExtractor
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer
from library.visualizers.MatplotlibHistogramVisualizer import MatplotlibHistogramVisualizer
from library.visualizers.MatplotlibVectorFieldVisualizer import MatplotlibVectorFieldVisualizer


def build_demo_pipeline_for_multi_object(video_path: str, initial_box=(300, 200, 80, 120), initial_barriers=None):
    """Create a runnable pipeline using only the public core components."""
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                path=video_path,
                config={"resize": (640, 480), "stride": 2, "max_frames": 420},
            )
        )
        .add_frame_cleaner(SmoothingFrameCleaner())
        .with_signal_extractor(OpenCVMultiObjectSignalExtractor(start_box=initial_box, max_objects=7, config={"show": True}))
        .add_analyzer(MultiObjectBarrierCountingAnalyzer(barriers=initial_barriers))
        .add_visualizer(MatplotlibHistogramVisualizer(config={"show": True}))
        .build_context()
    )
    return Pipeline(context)


def build_demo_pipeline_for_single_object(video_path: str, initial_box=(300, 200, 80, 120)):
    """Create a runnable pipeline using only the public core components."""
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                path=video_path,
                config={"resize": (640, 480), "stride": 2, "max_frames": 420},
            )
        )
        .add_frame_cleaner(OpenCVGrayFrameCleaner())
        .with_signal_extractor(OpenCVBufferedSignalExtractor(start_box=initial_box, config={"show": True}))
        .add_signal_cleaner(MovingAverageCleaner(window_size=5))
        .add_analyzer(VerticalFrequencyAnalyzer())
        .add_visualizer(MatplotlibFunctionVisualizer(config={"show": True}))
        .build_context()
    )
    return Pipeline(context)


def build_demo_pipeline_for_dense_optical_flow(video_path: str):
    """Create a runnable pipeline using only the public core components."""
    context = (
        FluentPipelineBuilder()
        .with_frame_extractor(
            OpenCVBufferedFrameExtractor(
                path=video_path,
                config={"resize": (640, 480), "stride": 2, "max_frames": 420},
            )
        )
        .with_signal_extractor(OpenCVDenseFarnebackSignalExtractor(config={"show": True}))
        .add_analyzer(DenseOpticalFlowVectorFieldAnalyzer())
        .add_visualizer(MatplotlibVectorFieldVisualizer(config={"show": True}))
        .build_context()
    )
    return Pipeline(context)


def main():
    root_dir = Path(__file__).resolve().parents[1]
    video_path = root_dir / "videos" / "Traffic.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    """ initial_box = OpenCVStartBoxSelector.select_start(str(video_path), (640, 480))
    initial_barriers = OpenCVBarrierSelector().select_barriers(str(video_path), ["pippo", "pluto", "paperino"]) """

    # TODO: mettere alcune librerie belle tipo: (vedi sotto)
    """
    Black: Formatter automatico: riscrive il codice per renderlo uniforme e leggibile senza dover pensare allo stile.
    Ruff: Linter veloce: trova errori, import inutili e cattive pratiche nel codice.
    mypy: Controllo dei tipi statico: verifica che i tipi delle variabili e funzioni siano coerenti.
    pytest: Framework di testing: permette di scrivere ed eseguire test in modo semplice e potente.
    MkDocs: Generatore di documentazione: crea un sito web leggibile partendo da file Markdown. Utile oltre ai docstrings e al README.md
    """

    # pipeline = build_demo_pipeline_for_multi_object(str(video_path), initial_box, initial_barriers)
    # pipeline = build_demo_pipeline_for_single_object(str(video_path), initial_box)
    pipeline = build_demo_pipeline_for_dense_optical_flow(str(video_path))
    pipeline.run()


if __name__ == "__main__":
    main()
