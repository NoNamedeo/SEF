from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.core import ConfigPipelineBuilder, Pipeline, PluginRegistry
from library.core.artifacts import BoxSignalSample, Frame, FrameBuffer, Signal, TwoDimGraphData
from library.core.interfaces import IAnalyzer, IData, IFrameExtractor, ISignal, ISignalExtractor, IVisualizer
from library.core.pipeline import CURRENT_PIPELINE_CONFIG_VERSION
from library.core.plugins import PluginCategory
from library.core.visualization import TextArtifact, VisualizationContext


class DemoFrameExtractor(IFrameExtractor):
    """Produce deterministic in-memory frames for documentation examples."""

    def __init__(self, frame_count: int = 3, config: dict | None = None) -> None:
        super().__init__(config)
        self.frame_count = int(frame_count)

    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(self.frame_count)
        for index in range(self.frame_count):
            buffer.put(
                Frame(
                    image=np.zeros((2, 2, 3), dtype=np.uint8),
                    index=index,
                    timestamp_seconds=float(index),
                )
            )
        buffer.close()
        return buffer


class DemoSignalExtractor(ISignalExtractor):
    """Convert each frame into one centroid sample."""

    def extract(self, buffer: FrameBuffer) -> ISignal:
        return Signal(
            [
                BoxSignalSample(
                    frame_index=frame.index or 0,
                    box=(0, 0, 2, 2),
                    centroid=(1.0, float(frame.index or 0)),
                    timestamp_seconds=frame.timestamp_seconds,
                )
                for frame in buffer
            ]
        )


class SampleCountAnalyzer(IAnalyzer):
    """Return a one-point graph containing the number of signal samples."""

    def analyze(self, signal: ISignal) -> TwoDimGraphData:
        count = len(list(signal))
        return TwoDimGraphData(
            x=[0.0],
            y=[float(count)],
            label="samples",
            title="Sample count",
        )


class SummaryVisualizer(IVisualizer):
    """Render a text summary artifact from the analyzer output."""

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[TextArtifact, ...]:
        sample_count = getattr(data, "y", [0.0])[0]
        return (
            TextArtifact(
                kind="text",
                title="Sample count summary",
                content=f"Sample count: {sample_count}",
                metadata={"pipeline_id": context.pipeline_id if context else None},
            ),
        )


def build_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(
        PluginCategory.FRAME_EXTRACTOR,
        "demo_frames",
        DemoFrameExtractor,
        "Produce deterministic documentation frames.",
        version="1.0.0",
    )
    registry.register(
        PluginCategory.SIGNAL_EXTRACTOR,
        "demo_signals",
        DemoSignalExtractor,
        "Convert documentation frames to centroid samples.",
        version="1.0.0",
    )
    registry.register(
        PluginCategory.ANALYZER,
        "sample_count",
        SampleCountAnalyzer,
        "Count signal samples.",
        version="1.0.0",
    )
    registry.register(
        PluginCategory.VISUALIZER,
        "summary_text",
        SummaryVisualizer,
        "Render sample count as text.",
        version="1.0.0",
    )
    return registry


def build_config(frame_count: int = 3) -> dict:
    return {
        "schema_version": CURRENT_PIPELINE_CONFIG_VERSION,
        "pipeline": {
            "frame_extractor": {
                "name": "demo_frames",
                "params": {"frame_count": frame_count},
            },
            "signal_extractor": {"name": "demo_signals"},
            "analyzers": [{"name": "sample_count"}],
            "visualizers": [{"name": "summary_text", "result_indices": [0]}],
        },
    }


def run_example(frame_count: int = 3):
    registry = build_registry()
    context = ConfigPipelineBuilder(registry).build_context(build_config(frame_count))
    return Pipeline(context, pipeline_id="docs-minimal-pipeline").run()


if __name__ == "__main__":
    outputs = run_example()
    print(f"results: {len(outputs.results)}")
    print(f"artifacts: {outputs.artifact_count}")
    print(f"sample_count: {outputs.results[0].y[0]}")
    print(f"summary: {outputs.final_artifacts[0].content}")
