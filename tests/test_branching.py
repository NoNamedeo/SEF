"""Tests for IBranchingRule, IEventEmitter, BranchingCoordinator, and lifecycle bus sharing."""
from __future__ import annotations

import unittest
from collections.abc import Iterable
from typing import Any

import numpy as np

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.IBranchingRule import IBranchingRule
from library.core.abstractions.IData import IData
from library.core.abstractions.IEventEmitter import IEventEmitter
from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.abstractions.IFrameExtractor import IFrameExtractor
from library.core.abstractions.ISignal import ISignal
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.Signal import Signal
from library.core.artifacts.SignalSample import SignalSample
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.events.DomainEvent import DomainEvent
from library.core.events.EventBus import EventBus
from library.core.events.PipelineLifecycleBus import (
    PipelineEvent,
    PipelineEventPayload,
    PipelineLifecycleBus,
)
from library.core.pipeline.BranchingCoordinator import BranchingCoordinator
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator


# ── Stub components ──────────────────────────────────────────────────────────


class StubFrameExtractor(IFrameExtractor):
    """Produces synthetic frames."""

    def __init__(self, num_frames: int = 4):
        super().__init__()
        self._num_frames = num_frames

    def extract(self, frame_cleaners: Iterable[IFrameCleaner]) -> FrameBuffer:
        buf = FrameBuffer(self._num_frames)
        for i in range(self._num_frames):
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            buf.put(Frame(image=img, index=i, timestamp_seconds=i * 0.1))
        buf.close()
        return buf


class StubSignalExtractor(ISignalExtractor):
    """Returns a fixed signal — no events."""

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples = [
            SignalSample(frame_index=i, box=(0, 0, 10, 10), centroid=(5.0, float(i)))
            for i, _ in enumerate(buffer)
        ]
        return Signal(samples)


class EmittingSignalExtractor(ISignalExtractor, IEventEmitter):
    """
    Signal extractor that emits a domain event mid-extraction.

    Emits ``"test_event"`` after processing the first frame.
    """

    def __init__(self, event_payload: dict[str, Any] | None = None):
        super().__init__()
        self._event_payload = event_payload or {"reason": "test"}

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples: list[SignalSample] = []
        for i, frame in enumerate(buffer):
            samples.append(
                SignalSample(
                    frame_index=i,
                    box=(0, 0, 10, 10),
                    centroid=(5.0, float(i)),
                )
            )
            if i == 0:
                self.emit("test_event", self._event_payload)
        return Signal(samples)


class StubAnalyzer(IAnalyzer):
    """Returns minimal TwoDimGraphData."""

    def analyze(self, signal: ISignal) -> IData:
        x = [float(s.frame_index) for s in signal]
        y = [float(s.centroid[1]) for s in signal if s.centroid]
        return TwoDimGraphData(x=x, y=y, label="stub", title="Stub Analysis")


# ── Branching rule stubs ─────────────────────────────────────────────────────


class AlwaysMatchRule(IBranchingRule):
    """Matches any event, builds a simple pipeline context."""

    def __init__(self):
        self._matched_events: list[DomainEvent] = []

    def matches(self, event: DomainEvent) -> bool:
        self._matched_events.append(event)
        return True

    def build_context(self, event: DomainEvent) -> PipelineContext:
        return PipelineContext(
            frame_extractor=StubFrameExtractor(num_frames=2),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer()],
        )


class SelectiveRule(IBranchingRule):
    """Matches only events of a specific type."""

    def __init__(self, target_event_type: str):
        self._target = target_event_type

    def matches(self, event: DomainEvent) -> bool:
        return event.event_type == self._target

    def build_context(self, event: DomainEvent) -> PipelineContext:
        return PipelineContext(
            frame_extractor=StubFrameExtractor(num_frames=2),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer()],
        )


class NeverMatchRule(IBranchingRule):
    """Never matches — for testing selective branching."""

    def matches(self, event: DomainEvent) -> bool:
        return False

    def build_context(self, event: DomainEvent) -> PipelineContext:
        raise AssertionError("Should never be called")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_context(
    signal_extractor: ISignalExtractor | None = None,
) -> PipelineContext:
    return PipelineContext(
        frame_extractor=StubFrameExtractor(),
        signal_extractor=signal_extractor or StubSignalExtractor(),
        analyzers=[StubAnalyzer()],
    )


def _make_orchestrator_with_branching(
    rules: list[IBranchingRule],
    signal_extractor: ISignalExtractor | None = None,
    lifecycle_bus: PipelineLifecycleBus | None = None,
) -> PipelineOrchestrator:
    """
    Build an orchestrator wired with a BranchingCoordinator.

    Explicit construction — no builder magic — makes the wiring visible.
    """
    event_bus = EventBus()
    branching = BranchingCoordinator(
        event_bus=event_bus,
        rules=rules,
        lifecycle_bus=lifecycle_bus,
    )
    return PipelineOrchestrator(
        _make_context(signal_extractor or EmittingSignalExtractor()),
        branching=branching,
        event_bus=event_bus,
        lifecycle_bus=lifecycle_bus,
    )


# ── Test classes ─────────────────────────────────────────────────────────────


class IEventEmitterTests(unittest.TestCase):
    """Verify IEventEmitter mixin behaviour."""

    def test_emit_without_bus_is_noop(self):
        extractor = EmittingSignalExtractor()
        # No bus injected — emit should silently do nothing
        extractor.emit("test_event", {"key": "value"})

    def test_emit_with_bus_publishes(self):
        extractor = EmittingSignalExtractor()
        bus = EventBus()
        extractor.event_bus = bus

        received: list[DomainEvent] = []
        bus.subscribe("test_event", received.append)

        extractor.emit("test_event", {"key": "value"})

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].event_type, "test_event")
        self.assertEqual(received[0].source, "EmittingSignalExtractor")
        self.assertEqual(received[0].payload["key"], "value")

    def test_event_bus_property(self):
        extractor = EmittingSignalExtractor()
        self.assertIsNone(extractor.event_bus)

        bus = EventBus()
        extractor.event_bus = bus
        self.assertIs(extractor.event_bus, bus)


class BranchingRuleTests(unittest.TestCase):
    """Verify IBranchingRule contract."""

    def test_always_match_rule(self):
        rule = AlwaysMatchRule()
        event = DomainEvent("any_event", "src")
        self.assertTrue(rule.matches(event))
        ctx = rule.build_context(event)
        self.assertIsNotNone(ctx.frame_extractor)

    def test_selective_rule_matches_target(self):
        rule = SelectiveRule("tracking_lost")
        self.assertTrue(rule.matches(DomainEvent("tracking_lost", "src")))
        self.assertFalse(rule.matches(DomainEvent("other_event", "src")))

    def test_never_match_rule(self):
        rule = NeverMatchRule()
        self.assertFalse(rule.matches(DomainEvent("any", "src")))


class BranchingCoordinatorTests(unittest.TestCase):
    """Verify BranchingCoordinator in isolation."""

    def test_coordinator_exposes_event_bus(self):
        bus = EventBus()
        coord = BranchingCoordinator(event_bus=bus, rules=[])
        self.assertIs(coord.event_bus, bus)
        coord.shutdown()

    def test_shutdown_is_idempotent(self):
        bus = EventBus()
        coord = BranchingCoordinator(event_bus=bus, rules=[])
        coord.shutdown()
        coord.shutdown()  # no error

    def test_collect_empty_when_no_events(self):
        bus = EventBus()
        coord = BranchingCoordinator(event_bus=bus, rules=[AlwaysMatchRule()])
        # No events published → no spawns
        self.assertEqual(coord.collect(timeout=1), [])
        coord.shutdown()

    def test_pending_count_starts_at_zero(self):
        bus = EventBus()
        coord = BranchingCoordinator(event_bus=bus, rules=[])
        self.assertEqual(coord.pending_count, 0)
        coord.shutdown()


class OrchestratorAutoSpawnTests(unittest.TestCase):
    """Verify that the orchestrator auto-spawns secondary pipelines on domain events."""

    def test_no_branching_no_bus(self):
        """Without branching, no EventBus is exposed."""
        orchestrator = PipelineOrchestrator(_make_context())
        self.assertIsNone(orchestrator.event_bus)
        results = orchestrator.run()
        self.assertEqual(len(results), 1)

    def test_branching_exposes_event_bus(self):
        """When BranchingCoordinator is configured, EventBus is accessible."""
        orchestrator = _make_orchestrator_with_branching(
            rules=[AlwaysMatchRule()],
        )
        self.assertIsNotNone(orchestrator.event_bus)
        orchestrator.shutdown()

    def test_auto_spawn_on_event(self):
        """EmittingSignalExtractor emits → AlwaysMatchRule spawns → secondary results available."""
        orchestrator = _make_orchestrator_with_branching(
            rules=[AlwaysMatchRule()],
        )

        primary = orchestrator.run()
        self.assertEqual(len(primary), 1)

        # The emitting SE fires "test_event" once → AlwaysMatchRule spawns 1 secondary
        secondary = orchestrator.collect_secondary_results(timeout=10)
        self.assertEqual(len(secondary), 1)
        self.assertEqual(len(secondary[0]), 1)  # 1 analyzer → 1 IData
        self.assertEqual(secondary[0][0].title, "Stub Analysis")

        orchestrator.shutdown()

    def test_selective_rule_ignores_non_matching(self):
        """SelectiveRule for 'other_type' should NOT spawn on 'test_event'."""
        orchestrator = _make_orchestrator_with_branching(
            rules=[SelectiveRule("other_type")],
        )

        orchestrator.run()

        secondary = orchestrator.collect_secondary_results(timeout=5)
        self.assertEqual(len(secondary), 0)

        orchestrator.shutdown()

    def test_multiple_rules_multiple_spawns(self):
        """Two rules that match → two secondary pipelines."""
        orchestrator = _make_orchestrator_with_branching(
            rules=[AlwaysMatchRule(), AlwaysMatchRule()],
        )

        orchestrator.run()

        secondary = orchestrator.collect_secondary_results(timeout=10)
        self.assertEqual(len(secondary), 2)

        orchestrator.shutdown()

    def test_pending_secondary_count_zero_without_branching(self):
        """pending_secondary_count is 0 when no coordinator configured."""
        orchestrator = PipelineOrchestrator(_make_context())
        self.assertEqual(orchestrator.pending_secondary_count, 0)

    def test_collect_empty_without_branching(self):
        """collect_secondary_results returns [] when no coordinator."""
        orchestrator = PipelineOrchestrator(_make_context())
        self.assertEqual(orchestrator.collect_secondary_results(), [])

    def test_shutdown_safe_without_branching(self):
        """shutdown is a no-op when no coordinator configured."""
        orchestrator = PipelineOrchestrator(_make_context())
        orchestrator.shutdown()  # no error


class LifecycleBusSharingTests(unittest.TestCase):
    """
    Verify that lifecycle events from secondary pipelines reach the
    primary's subscribers when a shared PipelineLifecycleBus is used.

    This is the core fix for Problem 2: lifecycle events were invisible
    on secondary pipelines before the PipelineLifecycleBus extraction.
    """

    def test_secondary_lifecycle_events_visible_on_shared_bus(self):
        """AFTER_RUN from the secondary pipeline arrives at the shared bus."""
        lifecycle_bus = PipelineLifecycleBus()
        payloads: list[PipelineEventPayload] = []
        lifecycle_bus.subscribe(PipelineEvent.AFTER_RUN, payloads.append)

        orchestrator = _make_orchestrator_with_branching(
            rules=[AlwaysMatchRule()],
            lifecycle_bus=lifecycle_bus,
        )

        orchestrator.run()
        orchestrator.collect_secondary_results(timeout=10)

        # At least 2 AFTER_RUN events: 1 primary + 1 secondary
        self.assertGreaterEqual(len(payloads), 2)

        orchestrator.shutdown()

    def test_before_run_events_from_both_pipelines(self):
        """BEFORE_RUN fires for primary AND secondary pipelines."""
        lifecycle_bus = PipelineLifecycleBus()
        before_runs: list[PipelineEventPayload] = []
        lifecycle_bus.subscribe(PipelineEvent.BEFORE_RUN, before_runs.append)

        orchestrator = _make_orchestrator_with_branching(
            rules=[AlwaysMatchRule()],
            lifecycle_bus=lifecycle_bus,
        )

        orchestrator.run()
        orchestrator.collect_secondary_results(timeout=10)

        # At least 2: primary + secondary
        self.assertGreaterEqual(len(before_runs), 2)

        orchestrator.shutdown()

    def test_no_shared_bus_means_secondary_events_isolated(self):
        """Without a shared bus, secondary lifecycle events are NOT visible."""
        primary_payloads: list[PipelineEventPayload] = []

        # Build WITHOUT a shared lifecycle_bus — each orchestrator gets its own
        orchestrator = _make_orchestrator_with_branching(
            rules=[AlwaysMatchRule()],
            lifecycle_bus=None,
        )
        # Subscribe on the primary's auto-created bus
        orchestrator.subscribe(PipelineEvent.AFTER_RUN, primary_payloads.append)

        orchestrator.run()
        orchestrator.collect_secondary_results(timeout=10)

        # Only the primary's AFTER_RUN — secondary used a different bus
        self.assertEqual(len(primary_payloads), 1)

        orchestrator.shutdown()


class FluentBuilderBranchingTests(unittest.TestCase):
    """Verify FluentPipelineBuilder assembles BranchingCoordinator correctly."""

    def test_fluent_builder_with_branching_rule(self):
        orchestrator = (
            FluentPipelineBuilder()
            .with_frame_extractor(StubFrameExtractor())
            .with_signal_extractor(EmittingSignalExtractor())
            .add_analyzer(StubAnalyzer())
            .add_branching_rule(AlwaysMatchRule())
            .build()
        )

        self.assertIsNotNone(orchestrator.event_bus)

        primary = orchestrator.run()
        secondary = orchestrator.collect_secondary_results(timeout=10)

        self.assertEqual(len(primary), 1)
        self.assertEqual(len(secondary), 1)

        orchestrator.shutdown()

    def test_fluent_builder_with_shared_lifecycle_bus(self):
        """Builder wires the shared lifecycle bus to both orchestrator and coordinator."""
        lifecycle_bus = PipelineLifecycleBus()
        payloads: list[PipelineEventPayload] = []
        lifecycle_bus.subscribe(PipelineEvent.AFTER_RUN, payloads.append)

        orchestrator = (
            FluentPipelineBuilder()
            .with_frame_extractor(StubFrameExtractor())
            .with_signal_extractor(EmittingSignalExtractor())
            .add_analyzer(StubAnalyzer())
            .add_branching_rule(AlwaysMatchRule())
            .with_lifecycle_bus(lifecycle_bus)
            .build()
        )

        orchestrator.run()
        orchestrator.collect_secondary_results(timeout=10)

        # Primary + secondary AFTER_RUN
        self.assertGreaterEqual(len(payloads), 2)

        orchestrator.shutdown()


if __name__ == "__main__":
    unittest.main()
