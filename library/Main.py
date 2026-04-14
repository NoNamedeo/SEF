from __future__ import annotations

from pathlib import Path

from library.analyzers.MultiObjectBarrierCountingAnalyzer import (
    MultiObjectBarrierCountingAnalyzer,
)
from library.core.pipeline import FluentPipelineBuilder
from library.core.pipeline.Pipeline import Pipeline
from library.core.utils.OpenCVBarrierSelector import OpenCVBarrierSelector
from library.core.utils.OpenCVStartBoxSelector import OpenCVStartBoxSelector
from library.frame_cleaners.SmoothingFrameCleaner import SmoothingFrameCleaner
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_extractors.OpenCVMultiObjectSignalExtractor import (
    OpenCVMultiObjectSignalExtractor,
)
from library.visualizers.MatplotlibHistogramVisualizer import MatplotlibHistogramVisualizer


def build_demo_pipeline(video_path: str, initial_box=(300, 200, 80, 120), initial_barriers=None):
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
        .with_signal_extractor(OpenCVMultiObjectSignalExtractor(start_box=initial_box, max_objects=2, config={"show": True}))
        .add_analyzer(MultiObjectBarrierCountingAnalyzer(barriers=initial_barriers))
        .build_context()
    )
    return Pipeline(context)


def main():
    root_dir = Path(__file__).resolve().parents[1]
    video_path = root_dir / "videos" / "Traffic.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    initial_box = OpenCVStartBoxSelector.select_start(str(video_path), (640, 480))
    initial_barriers = OpenCVBarrierSelector().select_barriers(str(video_path), ["a", "b", "c"])

    # TODO: mettere alcune librerie belle tipo: (vedi sotto)
    """
    Black: Formatter automatico: riscrive il codice per renderlo uniforme e leggibile senza dover pensare allo stile.
    Ruff: Linter veloce: trova errori, import inutili e cattive pratiche nel codice.
    mypy: Controllo dei tipi statico: verifica che i tipi delle variabili e funzioni siano coerenti.
    pytest: Framework di testing: permette di scrivere ed eseguire test in modo semplice e potente.
    MkDocs: Generatore di documentazione: crea un sito web leggibile partendo da file Markdown. Utile oltre ai docstrings e al README.md
    """

    pipeline = build_demo_pipeline(str(video_path), initial_box, initial_barriers)
    analysis_results = pipeline.run()

    visualizer = MatplotlibHistogramVisualizer(config={"show": True})
    visualizer.visualize(analysis_results[0])


if __name__ == "__main__":
    main()
