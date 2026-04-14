"""Tests verifying retry policy wiring in both builders."""

from __future__ import annotations

import unittest
from collections.abc import Iterable

import numpy as np

from library.core.interfaces.IAnalyzer import IAnalyzer
from library.core.interfaces.IData import IData
from library.core.interfaces.IFrameCleaner import IFrameCleaner
from library.core.interfaces.IFrameExtractor import IFrameExtractor
from library.core.interfaces.ISignal import ISignal
from library.core.interfaces.ISignalExtractor import ISignalExtractor
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.SignalSample import SignalSample
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.pipeline.ConfigPipelineBuilder import ConfigPipelineBuilder
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.plugins.PluginRegistry import PluginCategory, PluginRegistry
from library.retry_policies.ExponentialBackoffRetryPolicy import ExponentialBackoffRetryPolicy
from library.retry_policies.FixedRetryPolicy import FixedRetryPolicy
from library.retry_policies.NoRetryPolicy import NoRetryPolicy


# ── Stub components ──────────────────────────────────────────────────────────


class StubFrameExtractor(IFrameExtractor):
    def extract(self, frame_cleaners: Iterable[IFrameCleaner]) -> FrameBuffer:
        buf = FrameBuffer(2)
        for i in range(2):
            buf.put(Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), index=i))
        buf.close()
        return buf


class StubSignalExtractor(ISignalExtractor):
    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples = [
            SignalSample(frame_index=i, box=(0, 0, 4, 4), centroid=(2.0, float(i)))
            for i, _ in enumerate(buffer)
        ]
        return Signal(samples)


class StubAnalyzer(IAnalyzer):
    def analyze(self, signal: ISignal) -> IData:
        x = [float(s.frame_index) for s in signal]
        y = [float(s.centroid[1]) for s in signal if s.centroid]
        return TwoDimGraphData(x=x, y=y, label="stub", title="Stub")


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


class FluentBuilderRetryTests(unittest.TestCase):
    """
    Verify FluentPipelineBuilder retry policy wiring.

    Problem 3 fix: ``build()`` takes **no parameters**.
    Retry policy is configured via ``.with_max_retries(n)`` or
    ``.with_retry_policy(policy)`` — never through ``build()``.
    """

    def test_default_is_no_retry_policy(self):
        orchestrator = _base_builder().build()
        self.assertIsInstance(orchestrator._retry_policy, NoRetryPolicy)

    def test_with_max_retries_positive(self):
        orchestrator = _base_builder().with_max_retries(3).build()
        self.assertIsInstance(orchestrator._retry_policy, FixedRetryPolicy)

    def test_with_max_retries_zero(self):
        orchestrator = _base_builder().with_max_retries(0).build()
        self.assertIsInstance(orchestrator._retry_policy, NoRetryPolicy)

    def test_with_retry_policy_explicit(self):
        policy = ExponentialBackoffRetryPolicy(max_retries=2, base_delay=0.1)
        orchestrator = _base_builder().with_retry_policy(policy).build()
        self.assertIsInstance(orchestrator._retry_policy, ExponentialBackoffRetryPolicy)

    def test_last_call_wins(self):
        """with_max_retries then with_retry_policy — last call takes precedence."""
        policy = ExponentialBackoffRetryPolicy(max_retries=1, base_delay=0.1)
        orchestrator = _base_builder().with_max_retries(5).with_retry_policy(policy).build()
        self.assertIsInstance(orchestrator._retry_policy, ExponentialBackoffRetryPolicy)

    def test_build_takes_no_parameters(self):
        """build() signature has zero configuration parameters."""
        import inspect

        sig = inspect.signature(FluentPipelineBuilder.build)
        # Only 'self' — no other params
        params = [p for p in sig.parameters if p != "self"]
        self.assertEqual(params, [])

    def test_build_runs_successfully(self):
        """End-to-end: build + run must not crash."""
        orchestrator = _base_builder().with_max_retries(1).build()
        results = orchestrator.run()
        self.assertEqual(len(results), 1)


# ── Tests: ConfigPipelineBuilder ─────────────────────────────────────────────


class ConfigBuilderRetryTests(unittest.TestCase):
    """Verify ConfigPipelineBuilder correctly converts max_retries."""

    def _build_registry(self) -> PluginRegistry:
        registry = PluginRegistry()
        registry.register(PluginCategory.FRAME_EXTRACTOR, "stub_fe", StubFrameExtractor)
        registry.register(PluginCategory.SIGNAL_EXTRACTOR, "stub_se", StubSignalExtractor)
        registry.register(PluginCategory.ANALYZER, "stub_analyzer", StubAnalyzer)
        return registry

    def _build(self, max_retries: int = 0):
        config = {
            "pipeline": {
                "frame_extractor": {"name": "stub_fe"},
                "signal_extractor": {"name": "stub_se"},
                "analyzers": [{"name": "stub_analyzer"}],
                "orchestration": {"max_retries": max_retries},
            }
        }
        return ConfigPipelineBuilder(self._build_registry()).build(config)

    def test_zero_retries_uses_no_retry_policy(self):
        orchestrator = self._build(max_retries=0)
        self.assertIsInstance(orchestrator._retry_policy, NoRetryPolicy)

    def test_positive_retries_uses_fixed_policy(self):
        orchestrator = self._build(max_retries=2)
        self.assertIsInstance(orchestrator._retry_policy, FixedRetryPolicy)

    def test_missing_orchestration_defaults_to_no_retry(self):
        config = {
            "pipeline": {
                "frame_extractor": {"name": "stub_fe"},
                "signal_extractor": {"name": "stub_se"},
                "analyzers": [{"name": "stub_analyzer"}],
            }
        }
        orchestrator = ConfigPipelineBuilder(self._build_registry()).build(config)
        self.assertIsInstance(orchestrator._retry_policy, NoRetryPolicy)

    def test_build_runs_successfully(self):
        orchestrator = self._build(max_retries=1)
        results = orchestrator.run()
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
