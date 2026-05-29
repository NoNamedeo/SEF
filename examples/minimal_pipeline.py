from __future__ import annotations

import numpy as np

import sef
from library.core.artifacts import Frame, Signal
from library.core.artifacts.signal_sample import BoxSignalSample
from library.core.artifacts.buffer import FrameBuffer
from library.core.artifacts.data import TwoDimGraphData
from library.core.interfaces import IData, ISignal
from library.core.visualization import TextArtifact, VisualizationContext


class DemoFrameExtractor:
    """Produce deterministic in-memory frames for documentation examples."""

    def __init__(self, frame_count: int = 3, config: dict | None = None) -> None:
        self.config = config
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


class DemoSignalExtractor:
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


class SampleCountAnalyzer:
    """Return a one-point graph containing the number of signal samples."""

    def analyze(self, signal: ISignal) -> TwoDimGraphData:
        count = len(list(signal))
        return TwoDimGraphData(
            x=[0.0],
            y=[float(count)],
            label="samples",
            title="Sample count",
        )


class SummaryVisualizer:
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


def run_example(frame_count: int = 3):
    return (
        sef.pipeline("docs-minimal-pipeline")
        .frames(DemoFrameExtractor, frame_count=frame_count)
        .signals(DemoSignalExtractor)
        .analyze(SampleCountAnalyzer)
        .visualize(SummaryVisualizer)
        .run()
    )


if __name__ == "__main__":
    outputs = run_example()
    print(f"results: {len(outputs.results)}")
    print(f"artifacts: {outputs.artifact_count}")
    print(f"sample_count: {outputs.results[0].y[0]}")
    print(f"summary: {outputs.final_artifacts[0].content}")
