from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from typing import Any

import numpy as np

from library.core.artifacts.BoxSignalSample import BoxSignalSample
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalCleaner import ISignalCleaner
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.Pipeline import Pipeline
from library.core.pipeline.PipelineConfigExporter import PipelineConfigExporter
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry
from library.core.visualization.VisualArtifact import TextArtifact, VisualArtifact
from library.core.visualization.VisualizationContext import VisualizationContext


class ExportFrameExtractor(IFrameExtractor):
    """Deterministic frame source used by export round-trip tests."""

    def __init__(self, frame_count: int = 3, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.frame_count = int(frame_count)

    def extract(self) -> FrameBuffer:
        buffer = FrameBuffer(self.frame_count)
        for frame_index in range(self.frame_count):
            buffer.put(
                Frame(
                    image=np.zeros((4, 4, 3), dtype=np.uint8),
                    index=frame_index,
                    timestamp_seconds=frame_index * 0.5,
                    metadata={"source": "export-test"},
                )
            )
        buffer.close()
        return buffer


class ExportFrameCleaner(IFrameCleaner):
    """No-op cleaner with constructor state that must survive export."""

    def __init__(self, label: str = "clean", config: dict[str, Any] | None = None):
        super().__init__(config)
        self.label = label

    def clean(self, frame: Frame) -> Frame:
        frame.metadata["cleaner"] = self.label
        return frame


class ExportSignalExtractor(ISignalExtractor):
    """Build a simple centroid signal from frame indexes."""

    def __init__(self, offset: float = 0.0, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.offset = float(offset)

    def extract(self, buffer: FrameBuffer) -> ISignal:
        return Signal(
            [
                BoxSignalSample(
                    frame_index=frame.index or 0,
                    box=(0, 0, 1, 1),
                    centroid=(float(frame.index or 0), float(frame.index or 0) + self.offset),
                    timestamp_seconds=frame.timestamp_seconds,
                    metadata=dict(frame.metadata),
                )
                for frame in buffer
            ]
        )


class ExportSignalCleaner(ISignalCleaner):
    """Shift y-centroids to prove signal-cleaner params are exported."""

    def __init__(self, delta: float = 1.0, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.delta = float(delta)

    def clean(self, signal: ISignal) -> ISignal:
        cleaned = []
        for sample in signal:
            x, y = sample.centroid or (0.0, 0.0)
            cleaned.append(
                BoxSignalSample(
                    frame_index=sample.frame_index,
                    box=sample.box,
                    centroid=(x, y + self.delta),
                    timestamp_seconds=sample.timestamp_seconds,
                    metadata=dict(sample.metadata),
                )
            )
        return Signal(cleaned)


class ExportAnalyzer(IAnalyzer):
    """Return deterministic graph data for equivalence assertions."""

    def __init__(self, label: str = "exported", config: dict[str, Any] | None = None):
        super().__init__(config)
        self.label = label

    def analyze(self, signal: ISignal) -> IData:
        samples = list(signal)
        return TwoDimGraphData(
            x=[float(sample.frame_index) for sample in samples],
            y=[float(sample.centroid[1]) for sample in samples if sample.centroid is not None],
            label=self.label,
            title=f"{self.label} series",
            metadata={"sample_count": len(samples), "config": dict(self.config)},
        )


class ExportVisualizer(IVisualizer):
    """Return text artifacts so artifact metadata can be exported cheaply."""

    def __init__(self, style: str = "plain", config: dict[str, Any] | None = None):
        super().__init__(config)
        self.style = style

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        return (
            TextArtifact(
                kind="text",
                title=f"{self.style} artifact",
                content=f"{getattr(data, 'title', 'data')} via {self.style}",
                metadata={"pipeline_id": context.pipeline_id if context else None},
            ),
        )


def build_export_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(PluginCategory.FRAME_EXTRACTOR, "export_frame_extractor", ExportFrameExtractor)
    registry.register(PluginCategory.FRAME_CLEANER, "export_frame_cleaner", ExportFrameCleaner)
    registry.register(PluginCategory.SIGNAL_EXTRACTOR, "export_signal_extractor", ExportSignalExtractor)
    registry.register(PluginCategory.SIGNAL_CLEANER, "export_signal_cleaner", ExportSignalCleaner)
    registry.register(PluginCategory.ANALYZER, "export_analyzer", ExportAnalyzer)
    registry.register(PluginCategory.VISUALIZER, "export_visualizer", ExportVisualizer)
    return registry


def build_config() -> dict[str, Any]:
    return {
        "pipeline": {
            "frame_extractor": {
                "name": "export_frame_extractor",
                "params": {"frame_count": 4, "config": {"source_id": "unit-test"}},
            },
            "frame_cleaners": [
                {"name": "export_frame_cleaner", "params": {"label": "stable-cleaner"}},
            ],
            "signal_extractor": {
                "name": "export_signal_extractor",
                "params": {"offset": 2.5, "config": {"mode": "deterministic"}},
            },
            "signal_cleaners": [
                {"name": "export_signal_cleaner", "params": {"delta": 3.0}},
            ],
            "analyzers": [
                {"name": "export_analyzer", "params": {"label": "round-trip", "config": {"unit": "px"}}},
            ],
            "visualizers": [
                {"name": "export_visualizer", "params": {"style": "summary"}, "result_indices": [0]},
            ],
            "orchestration": {"max_retries": 0},
        }
    }


def context_signature(context) -> dict[str, Any]:
    return {
        "frame_extractor": {
            "frame_count": context.frame_extractor.frame_count,
            "config": dict(context.frame_extractor.config),
        },
        "frame_cleaners": [(cleaner.label, dict(cleaner.config)) for cleaner in context.frame_cleaners],
        "signal_extractor": {
            "offset": context.signal_extractor.offset,
            "config": dict(context.signal_extractor.config),
        },
        "signal_cleaners": [(cleaner.delta, dict(cleaner.config)) for cleaner in context.signal_cleaners],
        "analyzers": [(analyzer.label, dict(analyzer.config)) for analyzer in context.analyzers],
        "visualizer_bindings": [
            (binding.visualizer.style, tuple(binding.result_indices or ()))
            for binding in context.visualizer_bindings
        ],
    }


class PipelineExportTests(unittest.TestCase):
    def test_executed_pipeline_exports_config_code_and_metadata(self):
        registry = build_export_registry()
        original_context = ConfigPipelineBuilder(registry).build_context(build_config())

        outputs = Pipeline(
            original_context,
            pipeline_id="pipeline-export-test",
            execution_metadata={"request_id": "req-123"},
        ).run()

        reproducibility = outputs.metadata.reproducibility
        self.assertIn("config", reproducibility)
        self.assertIn("json", reproducibility)
        self.assertIn("yaml", reproducibility)
        self.assertIn("python_builder_code", reproducibility)

        exported_config = reproducibility["config"]
        self.assertEqual(exported_config["pipeline"]["frame_extractor"]["name"], "export_frame_extractor")
        self.assertEqual(exported_config["pipeline"]["signal_cleaners"][0]["params"]["delta"], 3.0)
        self.assertEqual(exported_config["pipeline"]["visualizers"][0]["result_indices"], [0])
        self.assertEqual(exported_config["execution"]["pipeline_id"], "pipeline-export-test")
        self.assertEqual(exported_config["execution"]["metadata"]["request_id"], "req-123")
        self.assertEqual(exported_config["artifacts"][0]["kind"], "text")
        self.assertEqual(exported_config["components"][0]["order"], 0)
        self.assertEqual(exported_config["components"][0]["registered_name"], "export_frame_extractor")

        json_export = json.loads(reproducibility["json"])
        self.assertEqual(json_export["pipeline"]["analyzers"][0]["name"], "export_analyzer")
        self.assertIn("pipeline:", reproducibility["yaml"])

        rebuilt_context = ConfigPipelineBuilder(registry).build_context(exported_config)
        self.assertEqual(context_signature(rebuilt_context), context_signature(original_context))
        self.assertEqual(Pipeline(rebuilt_context).run().results[0].y, outputs.results[0].y)

        namespace: dict[str, Any] = {}
        exec(reproducibility["python_builder_code"], namespace)
        code_context = namespace["build_context"](registry)
        self.assertEqual(context_signature(code_context), context_signature(original_context))

    def test_config_exporter_infers_fluent_component_params_with_registry(self):
        registry = build_export_registry()
        context = (
            FluentPipelineBuilder()
            .with_frame_extractor(ExportFrameExtractor(frame_count=2, config={"source_id": "fluent"}))
            .add_frame_cleaner(ExportFrameCleaner(label="fluent-cleaner"))
            .with_signal_extractor(ExportSignalExtractor(offset=1.25))
            .add_signal_cleaner(ExportSignalCleaner(delta=0.75))
            .add_analyzer(ExportAnalyzer(label="fluent", config={"unit": "px"}))
            .add_visualizer_for_results(ExportVisualizer(style="fluent-summary"), [0])
            .build_context()
        )

        def fixed_now() -> datetime:
            return datetime(2025, 1, 1, tzinfo=timezone.utc)

        exported_config = PipelineConfigExporter(registry, clock=fixed_now).export(context)

        self.assertEqual(exported_config["exported_at"], "2025-01-01T00:00:00+00:00")
        self.assertEqual(exported_config["pipeline"]["frame_extractor"]["name"], "export_frame_extractor")
        self.assertEqual(exported_config["pipeline"]["frame_extractor"]["params"]["frame_count"], 2)
        self.assertEqual(exported_config["pipeline"]["frame_cleaners"][0]["params"]["label"], "fluent-cleaner")
        self.assertEqual(exported_config["pipeline"]["signal_extractor"]["params"]["offset"], 1.25)
        self.assertEqual(exported_config["pipeline"]["signal_cleaners"][0]["params"]["delta"], 0.75)
        self.assertEqual(exported_config["pipeline"]["analyzers"][0]["params"]["config"]["unit"], "px")
        self.assertEqual(exported_config["pipeline"]["visualizers"][0]["params"]["style"], "fluent-summary")

        rebuilt_context = ConfigPipelineBuilder(registry).build_context(exported_config)
        self.assertEqual(context_signature(rebuilt_context), context_signature(context))
        self.assertNotIn("streamlit", sys.modules)


if __name__ == "__main__":
    unittest.main()
