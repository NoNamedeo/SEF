"""Tests verifying builders only create valid PipelineContext instances."""

from __future__ import annotations

import unittest

import numpy as np

from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.interfaces.IVisualizer import IVisualizer
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.BoxSignalSample import BoxSignalSample
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineErrors import PipelineConfigurationError
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry


# ── Stub components ──────────────────────────────────────────────────────────


class StubFrameExtractor(IFrameExtractor):
    def extract(self) -> FrameBuffer:
        buf = FrameBuffer(2)
        for i in range(2):
            buf.put(Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), index=i))
        buf.close()
        return buf


class StubSignalExtractor(ISignalExtractor):
    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples = [
            BoxSignalSample(frame_index=i, box=(0, 0, 4, 4), centroid=(2.0, float(i)))
            for i, _ in enumerate(buffer)
        ]
        return Signal(samples)


class StubAnalyzer(IAnalyzer):
    def analyze(self, signal: ISignal) -> IData:
        x = [float(s.frame_index) for s in signal]
        y = [float(s.centroid[1]) for s in signal if s.centroid]
        return TwoDimGraphData(x=x, y=y, label="stub", title="Stub")


class StubVisualizer(IVisualizer):
    def visualize(self, data: IData) -> None:
        return None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _base_builder() -> FluentPipelineBuilder:
    """Return a builder pre-loaded with stub components."""
    return (
        FluentPipelineBuilder()
        .with_frame_extractor(StubFrameExtractor())
        .with_signal_extractor(StubSignalExtractor())
        .add_analyzer(StubAnalyzer())
    )


# ── Tests: FluentPipelineBuilder ─────────────────────────────────────────────


class PipelineContextInvariantTests(unittest.TestCase):
    """PipelineContext must fail fast instead of allowing partial states."""

    def test_requires_frame_extractor(self):
        with self.assertRaisesRegex(ValueError, "frame_extractor"):
            PipelineContext(
                frame_extractor=None,
                signal_extractor=StubSignalExtractor(),
                analyzers=[StubAnalyzer()],
            )

    def test_requires_signal_extractor(self):
        with self.assertRaisesRegex(ValueError, "signal_extractor"):
            PipelineContext(
                frame_extractor=StubFrameExtractor(),
                signal_extractor=None,
                analyzers=[StubAnalyzer()],
            )

    def test_requires_at_least_one_analyzer(self):
        with self.assertRaisesRegex(ValueError, "at least one analyzer"):
            PipelineContext(
                frame_extractor=StubFrameExtractor(),
                signal_extractor=StubSignalExtractor(),
                analyzers=[],
            )

    def test_rejects_none_inside_component_collections(self):
        with self.assertRaisesRegex(ValueError, "cannot contain None"):
            PipelineContext(
                frame_extractor=StubFrameExtractor(),
                signal_extractor=StubSignalExtractor(),
                analyzers=[StubAnalyzer(), None],
            )

    def test_component_collections_are_immutable_snapshots(self):
        analyzers = [StubAnalyzer()]

        context = PipelineContext(
            frame_extractor=StubFrameExtractor(),
            signal_extractor=StubSignalExtractor(),
            analyzers=analyzers,
        )
        analyzers.append(StubAnalyzer())

        self.assertIsInstance(context.analyzers, tuple)
        self.assertEqual(len(context.analyzers), 1)


class FluentBuilderContextTests(unittest.TestCase):
    """Verify FluentPipelineBuilder has context-building responsibility only."""

    def test_build_context_returns_valid_context(self):
        context = _base_builder().build_context()

        self.assertIsInstance(context, PipelineContext)
        self.assertEqual(len(context.analyzers), 1)

    def test_builder_does_not_expose_runtime_methods(self):
        self.assertFalse(hasattr(FluentPipelineBuilder, "run"))
        self.assertFalse(hasattr(FluentPipelineBuilder, "build_pipeline"))
        self.assertFalse(hasattr(FluentPipelineBuilder, "build_orchestrator"))
        self.assertFalse(hasattr(FluentPipelineBuilder, "build_trigger_event"))

    def test_context_runs_through_orchestrator(self):
        context = _base_builder().build_context()

        results = PipelineOrchestrator().run(context)

        self.assertEqual(len(results), 1)


# ── Tests: ConfigPipelineBuilder ─────────────────────────────────────────────


class ConfigBuilderContextTests(unittest.TestCase):
    """Verify ConfigPipelineBuilder only converts configuration to context."""

    def _build_registry(self) -> PluginRegistry:
        registry = PluginRegistry()
        registry.register(PluginCategory.FRAME_EXTRACTOR, "stub_fe", StubFrameExtractor)
        registry.register(PluginCategory.SIGNAL_EXTRACTOR, "stub_se", StubSignalExtractor)
        registry.register(PluginCategory.ANALYZER, "stub_analyzer", StubAnalyzer)
        registry.register(PluginCategory.VISUALIZER, "stub_visualizer", StubVisualizer)
        return registry

    def _config(self) -> dict:
        return {
            "pipeline": {
                "frame_extractor": {"name": "stub_fe"},
                "signal_extractor": {"name": "stub_se"},
                "analyzers": [{"name": "stub_analyzer"}],
                "orchestration": {"max_retries": 2},
            }
        }

    def test_build_context_returns_valid_context(self):
        builder = ConfigPipelineBuilder(self._build_registry())
        context = builder.build_context(self._config())

        self.assertIsInstance(context, PipelineContext)
        self.assertEqual(len(context.analyzers), 1)

    def test_builder_does_not_expose_runtime_methods(self):
        self.assertFalse(hasattr(ConfigPipelineBuilder, "run"))
        self.assertFalse(hasattr(ConfigPipelineBuilder, "build_pipeline"))
        self.assertFalse(hasattr(ConfigPipelineBuilder, "build_orchestrator"))
        self.assertFalse(hasattr(ConfigPipelineBuilder, "build_trigger_event"))

    def test_context_runs_through_orchestrator(self):
        builder = ConfigPipelineBuilder(self._build_registry())
        context = builder.build_context(self._config())

        results = PipelineOrchestrator().run(context)

        self.assertEqual(len(results), 1)

    def test_missing_analyzers_fails_through_context_invariant(self):
        config = {
            "pipeline": {
                "frame_extractor": {"name": "stub_fe"},
                "signal_extractor": {"name": "stub_se"},
                "analyzers": [],
            }
        }
        builder = ConfigPipelineBuilder(self._build_registry())

        with self.assertRaisesRegex(PipelineConfigurationError, "at least one analyzer"):
            builder.build_context(config)

    def test_missing_pipeline_section_raises_configuration_error(self):
        builder = ConfigPipelineBuilder(self._build_registry())

        with self.assertRaisesRegex(PipelineConfigurationError, "pipeline"):
            builder.build_context({})

    def test_missing_plugin_name_raises_configuration_error_with_path(self):
        config = self._config()
        config["pipeline"]["frame_extractor"] = {}
        builder = ConfigPipelineBuilder(self._build_registry())

        with self.assertRaisesRegex(PipelineConfigurationError, "pipeline.frame_extractor.name"):
            builder.build_context(config)

    def test_unknown_plugin_raises_configuration_error(self):
        config = self._config()
        config["pipeline"]["analyzers"] = [{"name": "missing"}]
        builder = ConfigPipelineBuilder(self._build_registry())

        with self.assertRaisesRegex(PipelineConfigurationError, "Unknown plugin 'missing'"):
            builder.build_context(config)

    def test_invalid_plugin_params_raises_configuration_error(self):
        config = self._config()
        config["pipeline"]["frame_extractor"] = {
            "name": "stub_fe",
            "params": {"unexpected": True},
        }
        builder = ConfigPipelineBuilder(self._build_registry())

        with self.assertRaisesRegex(PipelineConfigurationError, "Invalid params"):
            builder.build_context(config)

    def test_config_visualizer_result_indices_build_selective_binding(self):
        config = self._config()
        config["pipeline"]["visualizers"] = [
            {"name": "stub_visualizer", "result_indices": [0]},
        ]
        builder = ConfigPipelineBuilder(self._build_registry())

        context = builder.build_context(config)

        self.assertEqual(len(context.visualizers), 0)
        self.assertEqual(len(context.visualizer_bindings), 1)
        self.assertEqual(context.visualizer_bindings[0].result_indices, (0,))

    def test_config_visualizer_rejects_invalid_result_indices(self):
        config = self._config()
        config["pipeline"]["visualizers"] = [
            {"name": "stub_visualizer", "result_indices": [-1]},
        ]
        builder = ConfigPipelineBuilder(self._build_registry())

        with self.assertRaisesRegex(PipelineConfigurationError, "result_indices"):
            builder.build_context(config)


if __name__ == "__main__":
    unittest.main()
