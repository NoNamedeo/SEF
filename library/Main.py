from __future__ import annotations

from pathlib import Path

from library.analyzers.HoriziontalPositionAnalyzer import HorizontalPositionAnalyzer
from library.analyzers.HorizontalFrequencyAnalyzer import HorizontalFrequencyAnalyzer
from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.analyzers.VerticalFrequencyAnalyzer import VerticalFrequencyAnalyzer
from library.core.pipeline.PipelineBuilder import PipelineBuilder
from library.core.utils.OpenCVStartBoxSelector import OpenCVStartBoxSelector
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.MovingAverageCleaner import MovingAverageCleaner
from library.signal_cleaners.SignalWidenerCleaner import SignalWidenerCleaner
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
                OpenCVBufferedSignalExtractor(tracker_type="CSRT", start_box=initial_box, config={"show": True})
            )
            .add_signal_cleaner(MovingAverageCleaner(window_size=5))
            .add_signal_cleaner(SignalWidenerCleaner(amplification=5))
            .add_analyzer(HorizontalPositionAnalyzer(config={"use_timestamps": True}))
            .add_analyzer(HorizontalFrequencyAnalyzer())
            .build()
    )


def main():
    root_dir = Path(__file__).resolve().parents[1]
    video_path = root_dir / "videos" / "Castle.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    initial_box = OpenCVStartBoxSelector.select_start(str(video_path), (640, 480))

    #TODO: aggiungere altri tracker (signalExtractor, tipo optical flow) con un signalsample specifico
    #TODO: aggiungere altri analyzers (tipo alcuni specifici per optical flow, altri per numero di macchine che vanno in una strada o in un altra, ecc)
    #TODO: aggiungere altri visualizers, frame extractors se ne vengono in mente altri
    #TODO: mettere alcune librerie fighe tipo: (vedi sotto)
    """
    Black: Formatter automatico: riscrive il codice per renderlo uniforme e leggibile senza dover pensare allo stile.
    Ruff: Linter veloce: trova errori, import inutili e cattive pratiche nel codice.
    mypy: Controllo dei tipi statico: verifica che i tipi delle variabili e funzioni siano coerenti.
    pytest: Framework di testing: permette di scrivere ed eseguire test in modo semplice e potente.
    MkDocs: Generatore di documentazione: crea un sito web leggibile partendo da file Markdown. Utile oltre ai docstrings e al README.md
    """

    pipeline = build_demo_pipeline(str(video_path), initial_box)
    analysis_results = pipeline.run()

    visualizer = MatplotlibFunctionVisualizer(config={"show": True})
    visualizer.visualize(analysis_results[0])


if __name__ == "__main__":
    main()
