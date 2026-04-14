"""Tests for IBranchingRule, IEventEmitter, BranchingCoordinator, and PipelineOrchestrator."""

from __future__ import annotations

import time
import unittest
from collections.abc import Iterable
from typing import Any

import numpy as np

from library.core.abstractions.IAnalyzer import IAnalyzer
from library.core.abstractions.IData import IData
from library.core.interfaces.IEventEmitter import IEventEmitter
from library.core.interfaces.pipeline.IBranchingRule import IBranchingRule
from library.core.abstractions.IFrameCleaner import IFrameCleaner
from library.core.abstractions.IFrameExtractor import IFrameExtractor
from library.core.abstractions.ISignal import ISignal
from library.core.abstractions.ISignalExtractor import ISignalExtractor
from library.core.artifacts.Frame import Frame
from library.core.artifacts.FrameBuffer import FrameBuffer
from library.core.artifacts.PipelineEvent import PipelineEvent
from library.core.artifacts.Signal import Signal
from library.core.artifacts.SignalSample import SignalSample
from library.core.artifacts.TwoDimGraphData import TwoDimGraphData
from library.core.events.DomainEvent import DomainEvent
from library.core.events.EventBus import EventBus
from library.core.events.PipelineLifecycleBus import (
    PipelineLifecycleEvent,
    PipelineLifecyclePayload,
)
from library.core.interfaces.pipeline.IEventBus import IEventBus
from library.core.interfaces.pipeline.IPipelineBuilder import IPipelineBuilder
from library.core.interfaces.pipeline.IPipelineMonitor import IPipelineMonitor
from library.core.interfaces.pipeline.IPipelineRunner import IPipelineRunner
from library.core.pipeline.BranchingCoordinator import BranchingCoordinator
from library.core.pipeline.DefaultPipelineBuilder import DefaultPipelineBuilder
from library.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from library.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor
from library.core.pipeline.Pipeline import Pipeline
from library.core.pipeline.PipelineContext import PipelineContext
from library.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from library.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner
from library.retry_policies.NoRetryPolicy import NoRetryPolicy


# ── Stub components ──────────────────────────────────────────────────────────


class StubFrameExtractor(IFrameExtractor):
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
    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples = [
            SignalSample(frame_index=i, box=(0, 0, 10, 10), centroid=(5.0, float(i)))
            for i, _ in enumerate(buffer)
        ]
        return Signal(samples)


class EmittingSignalExtractor(ISignalExtractor, IEventEmitter):
    """Emits ``"test_event"`` after processing the first frame."""

    def __init__(self, event_payload: dict[str, Any] | None = None):
        super().__init__()
        self._event_payload = event_payload or {"reason": "test"}

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples: list[SignalSample] = []
        for i, frame in enumerate(buffer):
            samples.append(
                SignalSample(frame_index=i, box=(0, 0, 10, 10), centroid=(5.0, float(i)))
            )
            if i == 0:
                self.emit("test_event", self._event_payload)
        return Signal(samples)


class StubAnalyzer(IAnalyzer):
    def analyze(self, signal: ISignal) -> IData:
        x = [float(s.frame_index) for s in signal]
        y = [float(s.centroid[1]) for s in signal if s.centroid]
        return TwoDimGraphData(x=x, y=y, label="stub", title="Stub Analysis")


# ── Branching rule stubs ─────────────────────────────────────────────────────


class AlwaysMatchRule(IBranchingRule):
    def matches(self, event: DomainEvent) -> bool:
        return True

    def build_context(self, event: DomainEvent) -> PipelineContext:
        return PipelineContext(
            frame_extractor=StubFrameExtractor(num_frames=2),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer()],
        )


class SelectiveRule(IBranchingRule):
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
    def matches(self, event: DomainEvent) -> bool:
        return False

    def build_context(self, event: DomainEvent) -> PipelineContext:
        raise AssertionError("Should never be called")


# ── Fake interface implementations for unit tests ────────────────────────────


class FakePipelineBuilder(IPipelineBuilder):
    def __init__(self) -> None:
        self.built: list[PipelineEvent] = []

    def build(self, event: PipelineEvent) -> Pipeline:
        self.built.append(event)
        return Pipeline(event.context)


class FakePipelineRunner(IPipelineRunner):
    def __init__(self) -> None:
        self.submitted: list[tuple[str, Pipeline]] = []
        self.cancelled: list[str] = []

    def submit(self, pipeline_id: str, pipeline: Pipeline) -> None:
        self.submitted.append((pipeline_id, pipeline))

    def cancel(self, pipeline_id: str) -> None:
        self.cancelled.append(pipeline_id)


class FakePipelineMonitor(IPipelineMonitor):
    def __init__(self) -> None:
        self._active: set[str] = set()
        self.completed: list[str] = []
        self.terminated: list[str] = []

    def register(self, pipeline_id: str) -> None:
        self._active.add(pipeline_id)

    def complete(self, pipeline_id: str) -> None:
        self._active.discard(pipeline_id)
        self.completed.append(pipeline_id)

    def terminate(self, pipeline_id: str) -> None:
        self._active.discard(pipeline_id)
        self.terminated.append(pipeline_id)

    def active_ids(self) -> list[str]:
        return list(self._active)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_context(signal_extractor: ISignalExtractor | None = None) -> PipelineContext:
    return PipelineContext(
        frame_extractor=StubFrameExtractor(),
        signal_extractor=signal_extractor or StubSignalExtractor(),
        analyzers=[StubAnalyzer()],
    )


def _make_orchestrator_with_branching(
    rules: list[IBranchingRule],
    signal_extractor: ISignalExtractor | None = None,
) -> tuple[EventBus, PipelineOrchestrator]:
    """Return (trigger_bus, orchestrator) wired with a BranchingCoordinator."""
    trigger_bus = EventBus()
    domain_bus = EventBus()
    monitor = InMemoryPipelineMonitor()
    runner = ThreadedPipelineRunner(
        monitor=monitor,
        retry_policy=NoRetryPolicy(),
        lifecycle_bus=trigger_bus,
    )
    pipeline_builder = DefaultPipelineBuilder(domain_bus=domain_bus)
    BranchingCoordinator(
        event_bus=domain_bus,
        rules=rules,
        trigger_bus=trigger_bus,
    )
    orchestrator = PipelineOrchestrator(
        builder=pipeline_builder,
        runner=runner,
        monitor=monitor,
        bus=trigger_bus,
    )
    context = _make_context(signal_extractor or EmittingSignalExtractor())
    trigger_bus.dispatch(PipelineEvent(pipeline_id="primary", context=context))
    return trigger_bus, orchestrator


def _wait_until_idle(orchestrator: PipelineOrchestrator, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while orchestrator.active_ids() and time.monotonic() < deadline:
        time.sleep(0.02)


# ── Tests ────────────────────────────────────────────────────────────────────


class IEventEmitterTests(unittest.TestCase):
    def test_emit_without_bus_is_noop(self):
        EmittingSignalExtractor().emit("test_event", {"key": "value"})

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
    def test_always_match_rule(self):
        rule = AlwaysMatchRule()
        event = DomainEvent("any_event", "src")
        self.assertTrue(rule.matches(event))
        self.assertIsNotNone(rule.build_context(event).frame_extractor)

    def test_selective_rule_matches_target(self):
        rule = SelectiveRule("tracking_lost")
        self.assertTrue(rule.matches(DomainEvent("tracking_lost", "src")))
        self.assertFalse(rule.matches(DomainEvent("other_event", "src")))

    def test_never_match_rule(self):
        self.assertFalse(NeverMatchRule().matches(DomainEvent("any", "src")))


class BranchingCoordinatorTests(unittest.TestCase):
    def test_exposes_event_bus(self):
        bus = EventBus()
        trigger = EventBus()
        coord = BranchingCoordinator(event_bus=bus, rules=[], trigger_bus=trigger)
        self.assertIs(coord.event_bus, bus)

    def test_no_dispatch_when_no_rules(self):
        domain_bus = EventBus()
        trigger_bus = EventBus()
        dispatched: list = []
        trigger_bus.subscribe(PipelineEvent.event_type, dispatched.append)

        BranchingCoordinator(event_bus=domain_bus, rules=[], trigger_bus=trigger_bus)
        domain_bus.dispatch(DomainEvent("any", "src"))

        self.assertEqual(dispatched, [])

    def test_dispatch_on_matching_rule(self):
        domain_bus = EventBus()
        trigger_bus = EventBus()
        dispatched: list[PipelineEvent] = []
        trigger_bus.subscribe(PipelineEvent.event_type, dispatched.append)

        BranchingCoordinator(
            event_bus=domain_bus, rules=[AlwaysMatchRule()], trigger_bus=trigger_bus
        )
        domain_bus.dispatch(DomainEvent("any", "src"))

        self.assertEqual(len(dispatched), 1)
        self.assertTrue(dispatched[0].pipeline_id.startswith("secondary-"))

    def test_no_dispatch_when_rule_does_not_match(self):
        domain_bus = EventBus()
        trigger_bus = EventBus()
        dispatched: list = []
        trigger_bus.subscribe(PipelineEvent.event_type, dispatched.append)

        BranchingCoordinator(
            event_bus=domain_bus,
            rules=[SelectiveRule("other")],
            trigger_bus=trigger_bus,
        )
        domain_bus.dispatch(DomainEvent("test_event", "src"))

        self.assertEqual(dispatched, [])


class PipelineOrchestratorUnitTests(unittest.TestCase):
    """Unit tests using fake interface implementations — no real threads."""

    def _make(
        self,
    ) -> tuple[
        EventBus,
        FakePipelineBuilder,
        FakePipelineRunner,
        FakePipelineMonitor,
        PipelineOrchestrator,
    ]:
        bus = EventBus()
        builder = FakePipelineBuilder()
        runner = FakePipelineRunner()
        monitor = FakePipelineMonitor()
        orch = PipelineOrchestrator(builder=builder, runner=runner, monitor=monitor, bus=bus)
        return bus, builder, runner, monitor, orch

    def test_pipeline_event_triggers_builder_monitor_runner(self):
        bus, builder, runner, monitor, orch = self._make()
        context = _make_context()
        bus.dispatch(PipelineEvent(pipeline_id="x", context=context))

        self.assertEqual(len(builder.built), 1)
        self.assertEqual(builder.built[0].pipeline_id, "x")
        self.assertEqual(len(runner.submitted), 1)
        self.assertEqual(runner.submitted[0][0], "x")
        self.assertIn("x", monitor._active)

    def test_active_ids_delegates_to_monitor(self):
        bus, _, _, monitor, orch = self._make()
        monitor.register("p1")
        monitor.register("p2")
        self.assertCountEqual(orch.active_ids(), ["p1", "p2"])

    def test_terminate_calls_runner_cancel_and_monitor_terminate(self):
        bus, _, runner, monitor, orch = self._make()
        context = _make_context()
        bus.dispatch(PipelineEvent(pipeline_id="p1", context=context))

        orch.terminate("p1")

        self.assertIn("p1", runner.cancelled)
        self.assertIn("p1", monitor.terminated)

    def test_multiple_events_each_trigger_chain(self):
        bus, builder, runner, monitor, orch = self._make()
        for i in range(3):
            bus.dispatch(PipelineEvent(pipeline_id=f"p{i}", context=_make_context()))

        self.assertEqual(len(builder.built), 3)
        self.assertEqual(len(runner.submitted), 3)


class PipelineOrchestratorIntegrationTests(unittest.TestCase):
    """Integration tests with real threads, real pipelines."""

    def test_pipeline_runs_and_emits_after_run(self):
        bus = EventBus()
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, lifecycle_bus=bus)
        builder = DefaultPipelineBuilder()
        orch = PipelineOrchestrator(builder=builder, runner=runner, monitor=monitor, bus=bus)

        payloads: list[PipelineLifecyclePayload] = []
        bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)

        context = _make_context()
        bus.dispatch(PipelineEvent(pipeline_id="run-1", context=context))

        _wait_until_idle(orch)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0].pipeline_id, "run-1")
        self.assertEqual(len(payloads[0].results), 1)

    def test_before_run_fires_before_after_run(self):
        bus = EventBus()
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, lifecycle_bus=bus)
        builder = DefaultPipelineBuilder()
        orch = PipelineOrchestrator(builder=builder, runner=runner, monitor=monitor, bus=bus)

        order: list[str] = []
        bus.subscribe(PipelineLifecycleEvent.BEFORE_RUN, lambda _: order.append("before"))
        bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, lambda _: order.append("after"))

        bus.dispatch(PipelineEvent(pipeline_id="x", context=_make_context()))
        _wait_until_idle(orch)

        self.assertEqual(order, ["before", "after"])

    def test_no_active_ids_after_completion(self):
        bus = EventBus()
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, lifecycle_bus=bus)
        builder = DefaultPipelineBuilder()
        orch = PipelineOrchestrator(builder=builder, runner=runner, monitor=monitor, bus=bus)

        bus.dispatch(PipelineEvent(pipeline_id="x", context=_make_context()))
        _wait_until_idle(orch)

        self.assertEqual(orch.active_ids(), [])


class OrchestratorBranchingTests(unittest.TestCase):
    def test_auto_spawn_on_domain_event(self):
        trigger_bus, orch = _make_orchestrator_with_branching(rules=[AlwaysMatchRule()])
        payloads: list[PipelineLifecyclePayload] = []
        trigger_bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)

        _wait_until_idle(orch, timeout=10)

        pipeline_ids = {p.pipeline_id for p in payloads}
        self.assertIn("primary", pipeline_ids)
        secondary_ids = {pid for pid in pipeline_ids if pid.startswith("secondary-")}
        self.assertEqual(len(secondary_ids), 1)

    def test_selective_rule_ignores_non_matching(self):
        trigger_bus, orch = _make_orchestrator_with_branching(
            rules=[SelectiveRule("other_event")]
        )
        payloads: list[PipelineLifecyclePayload] = []
        trigger_bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)

        _wait_until_idle(orch, timeout=10)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0].pipeline_id, "primary")

    def test_multiple_rules_spawn_multiple_secondaries(self):
        trigger_bus, orch = _make_orchestrator_with_branching(
            rules=[AlwaysMatchRule(), AlwaysMatchRule()]
        )
        payloads: list[PipelineLifecyclePayload] = []
        trigger_bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)

        _wait_until_idle(orch, timeout=10)

        secondary_ids = {
            p.pipeline_id for p in payloads if p.pipeline_id.startswith("secondary-")
        }
        self.assertEqual(len(secondary_ids), 2)

    def test_domain_events_do_not_reach_lifecycle_handlers(self):
        trigger_bus, orch = _make_orchestrator_with_branching(rules=[AlwaysMatchRule()])
        lifecycle_payloads: list = []
        trigger_bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, lifecycle_payloads.append)

        _wait_until_idle(orch, timeout=10)

        for p in lifecycle_payloads:
            self.assertIsInstance(p, PipelineLifecyclePayload)


class FluentBuilderBranchingTests(unittest.TestCase):
    def test_fluent_builder_with_branching_rule(self):
        bus = EventBus()
        orchestrator = (
            FluentPipelineBuilder()
            .with_trigger_bus(bus)
            .with_frame_extractor(StubFrameExtractor())
            .with_signal_extractor(EmittingSignalExtractor())
            .add_analyzer(StubAnalyzer())
            .add_branching_rule(AlwaysMatchRule())
            .build()
        )
        payloads: list[PipelineLifecyclePayload] = []
        bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)

        context = (
            FluentPipelineBuilder()
            .with_frame_extractor(StubFrameExtractor())
            .with_signal_extractor(EmittingSignalExtractor())
            .add_analyzer(StubAnalyzer())
            .build_context()
        )
        bus.dispatch(PipelineEvent(pipeline_id="primary", context=context))

        _wait_until_idle(orchestrator, timeout=10)

        pipeline_ids = {p.pipeline_id for p in payloads}
        self.assertIn("primary", pipeline_ids)
        self.assertTrue(any(pid.startswith("secondary-") for pid in pipeline_ids))

    def test_fluent_builder_build_context(self):
        builder = (
            FluentPipelineBuilder()
            .with_frame_extractor(StubFrameExtractor())
            .with_signal_extractor(StubSignalExtractor())
            .add_analyzer(StubAnalyzer())
        )
        context = builder.build_context()
        self.assertIsNotNone(context.frame_extractor)
        self.assertIsNotNone(context.signal_extractor)
        self.assertEqual(len(context.analyzers), 1)

    def test_fluent_builder_without_branching(self):
        bus = EventBus()
        orchestrator = FluentPipelineBuilder().with_trigger_bus(bus).build()
        payloads: list[PipelineLifecyclePayload] = []
        bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)

        context = _make_context()
        bus.dispatch(PipelineEvent(pipeline_id="solo", context=context))

        _wait_until_idle(orchestrator, timeout=10)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0].pipeline_id, "solo")


class InMemoryPipelineMonitorTests(unittest.TestCase):
    def test_register_and_active_ids(self):
        m = InMemoryPipelineMonitor()
        m.register("a")
        m.register("b")
        self.assertCountEqual(m.active_ids(), ["a", "b"])

    def test_complete_removes_from_active(self):
        m = InMemoryPipelineMonitor()
        m.register("a")
        m.complete("a")
        self.assertEqual(m.active_ids(), [])

    def test_terminate_removes_from_active(self):
        m = InMemoryPipelineMonitor()
        m.register("a")
        m.terminate("a")
        self.assertEqual(m.active_ids(), [])

    def test_complete_is_idempotent(self):
        m = InMemoryPipelineMonitor()
        m.register("a")
        m.complete("a")
        m.complete("a")
        self.assertEqual(m.active_ids(), [])


if __name__ == "__main__":
    unittest.main()
