from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from library.analyzers.HoriziontalPositionAnalyzer import HorizontalPositionAnalyzer
from library.analyzers.HorizontalFrequencyAnalyzer import HorizontalFrequencyAnalyzer
from library.analyzers.HorizontalVelocityAnalyzer import HorizontalVelocityAnalyzer
from library.analyzers.MultiObjectBarrierCountingAnalyzer import MultiObjectBarrierCountingAnalyzer
from library.analyzers.SparseOpticalFlowTrajectoryAnalyzer import SparseOpticalFlowTrajectoryAnalyzer
from library.analyzers.VerticalFrequencyAnalyzer import VerticalFrequencyAnalyzer
from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.analyzers.VerticalVelocityAnalyzer import VerticalVelocityAnalyzer
from library.core.pipeline.Pipeline import Pipeline
from library.core.pipeline.PipelineBuilder import PipelineBuilder
from library.core.utils.OpenCVBarrierSelector import OpenCVBarrierSelector
from library.core.utils.OpenCVStartBoxSelector import OpenCVStartBoxSelector
from library.frame_cleaners.OpenCVHistogramEqualizationFrameCleaner import (
    OpenCVHistogramEqualizationFrameCleaner,
)
from library.frame_cleaners.SmoothingFrameCleaner import SmoothingFrameCleaner
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.MovingAverageCleaner import MovingAverageCleaner
from library.signal_cleaners.OpticalFlowOutlierCleaner import OpticalFlowOutlierFilter

from library.signal_cleaners.OutlierRejectionCleaner import OutlierRejectionCleaner
from library.signal_cleaners.SignalWidenerCleaner import SignalWidenerCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.signal_extractors.OpenCVMultiObjectSignalExtractor import (
    OpenCVMultiObjectSignalExtractor,
)
from library.signal_extractors.OpenCVSparseOpticalFlowSignalExtractor import (
    OpenCVSparseOpticalFlowSignalExtractor,
)
from library.visualizers.MatplotlibTrajectoryVisualizer import MatplotlibTrajectoryVisualizer
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer

from library.visualizers.MatplotlibHistogramVisualizer import (
    MatplotlibHistogramVisualizer,
)


VIDEO_SIZE = (960, 540)
FRAME_EXTRACTOR_CONFIG = {"resize": VIDEO_SIZE, "stride": 2, "max_frames": 360}
DEFAULT_START_BOX = (430, 220, 160, 110)
SHOW_PLOTS = True
SHOW_OPENCV_WINDOWS = True
DEFAULT_BARRIER_NAMES = ("left_gate", "center_gate", "right_gate")


@dataclass(slots=True)
class PipelineScenario:
    name: str
    pipeline: Pipeline
    visualizer: Any
    description: str


def build_frame_extractor(video_path: str, *, max_frames: int | None = None) -> OpenCVBufferedFrameExtractor:
    config = dict(FRAME_EXTRACTOR_CONFIG)
    if max_frames is not None:
        config["max_frames"] = max_frames
    return OpenCVBufferedFrameExtractor(path=video_path, config=config)


def build_function_visualizer() -> MatplotlibFunctionVisualizer:
    return MatplotlibFunctionVisualizer(
        config={
            "show": False,
            "grid": True,
            "show_scatter": True,
            "figure_size": (11, 6),
        }
    )


def resolve_start_box(video_path: Path) -> tuple[int, int, int, int]:
    try:
        return OpenCVStartBoxSelector.select_start(str(video_path), VIDEO_SIZE)
    except Exception as exc:
        print(
            "ROI non selezionata, uso la box di fallback "
            f"{DEFAULT_START_BOX}. Motivo: {exc}"
        )
        return DEFAULT_START_BOX


def build_multi_object_scenarios(video_path: str, start_box: tuple[int, int, int, int]) -> list[PipelineScenario]:
    barriers = resolve_barriers(video_path)

    pipeline = (
        PipelineBuilder()
        .with_frame_extractor(build_frame_extractor(video_path, max_frames=280))
        .add_frame_cleaner(SmoothingFrameCleaner(alpha=0.8))
        .with_signal_extractor(
            OpenCVMultiObjectSignalExtractor(
                tracker_type="CSRT",
                start_box=start_box,
                max_objects=12,
                similarity_threshold=0.55,
                config={"show": SHOW_OPENCV_WINDOWS},
            )
        )
        .add_analyzer(MultiObjectBarrierCountingAnalyzer(barriers=barriers))
        .build()
    )

    return [
        PipelineScenario(
            name="multi_object_barrier_counting",
            pipeline=pipeline,
            visualizer=MatplotlibHistogramVisualizer(config={"show": False}),
            description=(
                "OpenCVMultiObjectSignalExtractor + MultiObjectBarrierCountingAnalyzer "
                "+ MatplotlibHistogramVisualizer su Traffic.mp4"
            ),
        )
    ]


def build_optical_flow_scenarios(video_path: str, start_box: tuple[int, int, int, int]) -> list[PipelineScenario]:
    scenarios: list[PipelineScenario] = []

    optical_flow_cases = [
        (
            "optical_flow_horizontal_position",
            [HorizontalPositionAnalyzer(config={"use_timestamps": True})],
            [],
            build_function_visualizer(),
            "Posizione orizzontale del centro ROI tramite optical flow",
        ),
        (
            "optical_flow_vertical_position",
            [VerticalPositionAnalyzer(config={"use_timestamps": True})],
            [],
            build_function_visualizer(),
            "Posizione verticale del centro ROI tramite optical flow",
        ),
        (
            "optical_flow_horizontal_velocity",
            [HorizontalVelocityAnalyzer(config={"use_timestamps": True})],
            [OpticalFlowOutlierFilter(threshold=3.0)],
            build_function_visualizer(),
            "Velocita orizzontale con rimozione spike sul vettore di moto",
        ),
        (
            "optical_flow_vertical_velocity",
            [VerticalVelocityAnalyzer(config={"use_timestamps": True})],
            [OpticalFlowOutlierFilter(threshold=3.0)],
            build_function_visualizer(),
            "Velocita verticale con rimozione spike sul vettore di moto",
        ),
        (
            "optical_flow_horizontal_frequency",
            [HorizontalFrequencyAnalyzer()],
            [OpticalFlowOutlierFilter(threshold=2.5)],
            build_function_visualizer(),
            "Spettro in frequenza della traiettoria orizzontale dell'optical flow",
        ),
        (
            "optical_flow_vertical_frequency",
            [VerticalFrequencyAnalyzer()],
            [OpticalFlowOutlierFilter(threshold=2.5)],
            build_function_visualizer(),
            "Spettro in frequenza della traiettoria verticale dell'optical flow",
        ),
        (
            "optical_flow_vector_field",
            [SparseOpticalFlowTrajectoryAnalyzer()],
            [OpticalFlowOutlierFilter(threshold=2.5)],
            MatplotlibTrajectoryVisualizer(config={"show": False, "scale": 1.0}),
            "Campo vettoriale medio dell'optical flow",
        ),
        (
            "optical_flow_heatmap",
            [SparseOpticalFlowTrajectoryAnalyzer()],
            [OpticalFlowOutlierFilter(threshold=2.5)],
            MatplotlibTrajectoryVisualizer(config={"show": False, "grid_size": 24}),
            "Heatmap dell'intensita del moto per optical flow",
        ),
    ]

    for name, analyzers, signal_cleaners, visualizer, description in optical_flow_cases:
        pipeline_builder = (
            PipelineBuilder()
            .with_frame_extractor(build_frame_extractor(video_path, max_frames=260))
            .add_frame_cleaner(OpenCVHistogramEqualizationFrameCleaner())
            .add_frame_cleaner(SmoothingFrameCleaner(alpha=0.75))
            .with_signal_extractor(
                OpenCVSparseOpticalFlowSignalExtractor(example_box=start_box, max_corners=150, quality_level=0.2,
                                                       min_distance=5, block_size=7,
                                                       config={"show": SHOW_OPENCV_WINDOWS})
            )
            .with_signal_cleaners(signal_cleaners)
            .with_analyzers(analyzers)
        )

        scenarios.append(
            PipelineScenario(
                name=name,
                pipeline=pipeline_builder.build(),
                visualizer=visualizer,
                description=description,
            )
        )

    return scenarios


def build_buffered_scenarios(video_path: str, start_box: tuple[int, int, int, int]) -> list[PipelineScenario]:
    scenarios: list[PipelineScenario] = []

    buffered_cases = [
        (
            "buffered_horizontal_position_raw",
            [HorizontalPositionAnalyzer(config={"use_timestamps": True})],
            [],
            "Tracking raw della posizione orizzontale",
        ),
        (
            "buffered_vertical_position_smoothed",
            [VerticalPositionAnalyzer(config={"use_timestamps": True})],
            [MovingAverageCleaner(window_size=5)],
            "Tracking verticale con smoothing della traiettoria",
        ),
        (
            "buffered_horizontal_velocity_smoothed",
            [HorizontalVelocityAnalyzer(config={"use_timestamps": True})],
            [MovingAverageCleaner(window_size=7)],
            "Velocita orizzontale dopo moving average",
        ),
        (
            "buffered_vertical_velocity_outlier_replaced",
            [VerticalVelocityAnalyzer(config={"use_timestamps": True})],
            [OutlierRejectionCleaner(threshold=3.0, mode="replace"), MovingAverageCleaner(window_size=5)],
            "Velocita verticale con outlier rejection + smoothing",
        ),
        (
            "buffered_horizontal_frequency_widened",
            [HorizontalFrequencyAnalyzer()],
            [MovingAverageCleaner(window_size=5), SignalWidenerCleaner(amplification=3.0)],
            "Spettro orizzontale con segnale amplificato",
        ),
        (
            "buffered_vertical_frequency_cleaned",
            [VerticalFrequencyAnalyzer()],
            [OutlierRejectionCleaner(threshold=2.8, mode="clip"), MovingAverageCleaner(window_size=5)],
            "Spettro verticale con clipping degli outlier",
        ),
        (
            "buffered_horizontal_position_full_cleaning",
            [HorizontalPositionAnalyzer(config={"use_timestamps": True})],
            [
                OutlierRejectionCleaner(threshold=3.0, mode="replace"),
                MovingAverageCleaner(window_size=5),
                SignalWidenerCleaner(amplification=2.0),
            ],
            "Posizione orizzontale con catena completa di cleaner",
        ),
    ]

    for name, analyzers, signal_cleaners, description in buffered_cases:
        pipeline_builder = (
            PipelineBuilder()
            .with_frame_extractor(build_frame_extractor(video_path, max_frames=280))
            .add_frame_cleaner(OpenCVHistogramEqualizationFrameCleaner())
            .add_frame_cleaner(SmoothingFrameCleaner(alpha=0.8))
            .with_signal_extractor(
                OpenCVBufferedSignalExtractor(
                    tracker_type="CSRT",
                    start_box=start_box,
                    config={"show": SHOW_OPENCV_WINDOWS},
                )
            )
            .with_signal_cleaners(signal_cleaners)
            .with_analyzers(analyzers)
        )

        scenarios.append(
            PipelineScenario(
                name=name,
                pipeline=pipeline_builder.build(),
                visualizer=build_function_visualizer(),
                description=description,
            )
        )

    return scenarios


def print_result_summary(scenario: PipelineScenario, result: Any) -> None:
    metadata = getattr(result, "metadata", {}) or {}
    print(f"[OK] {scenario.name}")
    print(f"      {scenario.description}")
    if metadata:
        print(f"      metadata: {metadata}")


def build_default_barriers() -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    width, height = VIDEO_SIZE
    return {
        "left_gate": ((int(width * 0.35), 0), (int(width * 0.35), height)),
        "center_gate": ((int(width * 0.50), 0), (int(width * 0.50), height)),
        "right_gate": ((int(width * 0.65), 0), (int(width * 0.65), height)),
    }


def resolve_barriers(video_path: str) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    try:
        return OpenCVBarrierSelector.select_barriers(
            video_path=video_path,
            barrier_names=DEFAULT_BARRIER_NAMES,
            resize=VIDEO_SIZE,
        )
    except Exception as exc:
        barriers = build_default_barriers()
        print(
            "Barriere non selezionate, uso quelle di fallback "
            f"{barriers}. Motivo: {exc}"
        )
        return barriers


def run_scenarios(scenarios: list[PipelineScenario]) -> None:
    for scenario in scenarios:
        print(f"\n=== {scenario.name} ===")
        results = scenario.pipeline.run()

        if not results:
            raise ValueError(f"Nessun risultato prodotto dalla pipeline {scenario.name}")

        result = results[0]
        scenario.visualizer.visualize(result)
        print_result_summary(scenario, result)


def build_all_scenarios(video_path: str, start_box: tuple[int, int, int, int]) -> list[PipelineScenario]:
    return [
        *build_multi_object_scenarios(video_path, start_box),
        *build_optical_flow_scenarios(video_path, start_box),
        *build_buffered_scenarios(video_path, start_box),
    ]


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    video_path = root_dir / "videos" / "Crowd.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    start_box = resolve_start_box(video_path)
    print(f"Start box selezionata: {start_box}")
    print("Preview real-time degli extractor attiva. Premi ESC per chiudere la finestra corrente.")

    scenarios = build_all_scenarios(str(video_path), start_box)
    print(f"Pipeline preparate: {len(scenarios)}")

    run_scenarios(scenarios)

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
