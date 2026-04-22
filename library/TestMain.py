from __future__ import annotations

from pathlib import Path

from library.analyzers.HoriziontalPositionAnalyzer import HorizontalPositionAnalyzer
from library.analyzers.HorizontalFrequencyAnalyzer import HorizontalFrequencyAnalyzer
from library.analyzers.HorizontalVelocityAnalyzer import HorizontalVelocityAnalyzer
from library.analyzers.MultipleDistanceAnalyzer import MultipleDistanceAnalyzer
from library.analyzers.VerticalFrequencyAnalyzer import VerticalFrequencyAnalyzer
from library.analyzers.VerticalPositionAnalyzer import VerticalPositionAnalyzer
from library.analyzers.VerticalVelocityAnalyzer import VerticalVelocityAnalyzer
from library.core.artifacts.BoxSignalSample import BoxSignalSample
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.events.Event import Event
from library.core.events.EventBus import EventBus
from library.core.events.PipelineEvent import PipelineEvent
from library.core.events.PipelineLifecycleEvent import PipelineLifecycleEvent
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry
from library.core.utils.OpenCVMultiStartBoxSelector import OpenCVMultiStartBoxSelector
from library.core.visualization.PipelineOutputs import PipelineOutputs
from library.core.visualization.VisualArtifact import TextArtifact, VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext
from library.frame_cleaners.OpenCVBackgroundSubtractionFrameCleaner import OpenCVBackgroundSubtractionFrameCleaner
from library.frame_cleaners.OpenCVGrayFrameCleaner import OpenCVGrayFrameCleaner
from library.frame_cleaners.OpenCVHistogramEqualizationFrameCleaner import OpenCVHistogramEqualizationFrameCleaner
from library.frame_cleaners.OpenCVResizeFrameCleaner import OpenCVResizeFrameCleaner
from library.frame_cleaners.OpenCVZoomFrameCleaner import OpenCVZoomFrameCleaner
from library.frame_cleaners.SmoothingFrameCleaner import SmoothingFrameCleaner
from library.frame_extractors.OpenCVBufferedFrameExtractor import OpenCVBufferedFrameExtractor
from library.signal_cleaners.MovingAverageCleaner import MovingAverageCleaner
from library.signal_extractors.OpenCVBufferedSignalExtractor import OpenCVBufferedSignalExtractor
from library.core.utils.OpenCVStartBoxSelector import OpenCVStartBoxSelector
from library.signal_extractors.OpenCVMultiManualSignalExtractor import OpenCVMultiManualSignalExtractor
from library.signal_extractors.SAM2SingleFigureSignalExtractor import SAM2SingleFigureSignalExtractor
from library.visualizers.MatplotlibFunctionVisualizer import MatplotlibFunctionVisualizer
from library.visualizers.MatplotlibHistogramVisualizer import MatplotlibHistogramVisualizer
from library.visualizers.MatplotlibTrajectoryVisualizer import MatplotlibTrajectoryVisualizer
from PIL import Image
import io

def build_fluent_context(video_path, zoom_box, start_box, start_boxes, resize) -> PipelineContext:
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(OpenCVBufferedFrameExtractor(
            video_path,
            config={
                "max_frames": 480 # (480 Frame)/(24 FPS) = 20 secondi
            }
        ))
        .add_frame_cleaner(OpenCVResizeFrameCleaner(resize))
        #.add_frame_cleaner(OpenCVZoomFrameCleaner(zoom_box))
        #.add_frame_cleaner(OpenCVHistogramEqualizationFrameCleaner())
        .with_signal_extractor(SAM2SingleFigureSignalExtractor(
            start_box=start_box,
            config={
                "show": True
            }
        ))
        .with_analyzers([VerticalPositionAnalyzer()])
        .add_visualizer_for_results(MatplotlibFunctionVisualizer(), [0])
        .build_context()
    )

def main():
    #TODO in caso disinstalla ultralytics

    BASE_DIR = Path(__file__).resolve().parent
    video_path = BASE_DIR.parent / "videos" / "Baloons.mp4"

    resize = (800, 600)

    zoom_box = None
    #zoom_box = OpenCVStartBoxSelector().select_start(str(video_path), resize)

    start_box = None
    start_box = OpenCVStartBoxSelector().select_start(str(video_path), frame_cleaners=[OpenCVResizeFrameCleaner(resize)])
    #number_of_boxes = int(input("How many boxes would you like?: "))
    start_boxes = None
    #start_boxes = OpenCVMultiStartBoxSelector().select_start(str(video_path), number_of_boxes, frame_cleaners=[OpenCVResizeFrameCleaner(resize), OpenCVZoomFrameCleaner(zoom_box)])

    orchestrator = PipelineOrchestrator()
    try:
        outputs = orchestrator.run(
            build_fluent_context(video_path, zoom_box, start_box, start_boxes, resize),
            pipeline_id="2",
        )
    finally:
        orchestrator.shutdown()

    for image_artifact in outputs.artifacts:
        image = Image.open(io.BytesIO(image_artifact.data))
        image.show()

if __name__ == "__main__":
    main()
