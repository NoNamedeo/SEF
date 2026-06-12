"""Tests for IBranchingRule, IEventEmitter, BranchingCoordinator, and PipelineOrchestrator."""

from __future__ import annotations

import time
import unittest
from concurrent.futures import Future
from threading import Event as ThreadingEvent
from typing import Any

import numpy as np

from sef.core.pipeline.NoRetryPolicy import NoRetryPolicy
from sef.core.artifacts.buffer.FrameBuffer import FrameBuffer
from sef.core.artifacts.data.TwoDimGraphData import TwoDimGraphData
from sef.core.artifacts.Frame import Frame
from sef.core.artifacts.Signal import Signal
from sef.core.artifacts.signal_sample.BoxSignalSample import BoxSignalSample
from sef.core.events.Event import Event
from sef.core.events.EventBus import EventBus
from sef.core.events.PipelineEvent import PipelineEvent
from sef.core.events.PipelineLifecycleEvent import (
    PipelineLifecycleEvent,
)
from sef.core.interfaces.IAnalyzer import IAnalyzer
from sef.core.interfaces.IData import IData
from sef.core.interfaces.IEventEmitter import IEventEmitter
from sef.core.interfaces.IFrameExtractor import IFrameExtractor
from sef.core.interfaces.ISignal import ISignal
from sef.core.interfaces.ISignalExtractor import ISignalExtractor
from sef.core.interfaces.IVisualizer import IVisualizer
from sef.core.interfaces.pipeline.IBranchingRule import IBranchingRule
from sef.core.interfaces.pipeline.IEventBus import IEventBus
from sef.core.interfaces.pipeline.IPipelineFactory import IPipelineFactory
from sef.core.interfaces.pipeline.IPipelineMonitor import IPipelineMonitor
from sef.core.interfaces.pipeline.IPipelineRunner import IPipelineRunner
from sef.core.pipeline.BranchingCoordinator import BranchingCoordinator
from sef.core.pipeline.FluentPipelineBuilder import FluentPipelineBuilder
from sef.core.pipeline.InMemoryPipelineMonitor import InMemoryPipelineMonitor
from sef.core.pipeline.Pipeline import Pipeline, PipelineExecutionError
from sef.core.pipeline.PipelineContext import PipelineContext
from sef.core.pipeline.PipelineErrors import PipelineRunAlreadyActiveError
from sef.core.pipeline.PipelineOrchestrator import PipelineOrchestrator
from sef.core.pipeline.PipelineRunSnapshot import PipelineRunSnapshot, PipelineRunState
from sef.core.pipeline.ThreadedPipelineRunner import ThreadedPipelineRunner
from sef.core.pipeline.VisualizerBinding import VisualizerBinding
from sef.core.visualization.PipelineOutputs import PipelineOutputs
from sef.core.visualization.VisualArtifact import VisualArtifact
from sef.core.visualization.VisualizationContext import VisualizationContext

# ── Stub components ──────────────────────────────────────────────────────────


class StubFrameExtractor(IFrameExtractor):
    def __init__(self, num_frames: int = 4):
        super().__init__()
        self._num_frames = num_frames

    def extract(self) -> FrameBuffer:
        buf = FrameBuffer(self._num_frames)
        for i in range(self._num_frames):
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            buf.put(Frame(image=img, index=i, timestamp_seconds=i * 0.1))
        buf.close()
        return buf


class BlockingFrameExtractor(IFrameExtractor):
    def __init__(self, started: ThreadingEvent, release: ThreadingEvent) -> None:
        super().__init__()
        self._started = started
        self._release = release

    def extract(self) -> FrameBuffer:
        self._started.set()
        self._release.wait(timeout=5)
        return StubFrameExtractor(num_frames=1).extract()


class StubSignalExtractor(ISignalExtractor):
    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples = [BoxSignalSample(frame_index=i, box=(0, 0, 10, 10), centroid=(5.0, float(i))) for i, _ in enumerate(buffer)]
        return Signal(samples)


class EmittingSignalExtractor(ISignalExtractor, IEventEmitter):
    """Emits ``"test_event"`` after processing the first frame."""

    def __init__(self, event_payload: dict[str, Any] | None = None):
        super().__init__()
        self._event_payload = event_payload or {"reason": "test"}

    def extract(self, buffer: FrameBuffer) -> ISignal:
        samples: list[BoxSignalSample] = []
        for i, _frame in enumerate(buffer):
            samples.append(BoxSignalSample(frame_index=i, box=(0, 0, 10, 10), centroid=(5.0, float(i))))
            if i == 0:
                self.emit("test_event", self._event_payload)
        return Signal(samples)


class StubAnalyzer(IAnalyzer):
    def analyze(self, signal: ISignal) -> IData:
        x = [float(s.frame_index) for s in signal]
        y = [float(s.centroid[1]) for s in signal if s.centroid]
        return TwoDimGraphData(x=x, y=y, label="stub", title="Stub Analysis")


class OtherAnalyzer(IAnalyzer):
    def analyze(self, signal: ISignal) -> IData:
        x = [float(s.frame_index) for s in signal]
        y = [float(s.frame_index * 2) for s in signal]
        return TwoDimGraphData(x=x, y=y, label="other", title="Other Analysis")


class FailingAnalyzer(IAnalyzer):
    def analyze(self, signal: ISignal) -> IData:
        raise RuntimeError("analysis failed")


class RecordingVisualizer(IVisualizer):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def render(
        self,
        data: IData,
        context: VisualizationContext | None = None,
    ) -> tuple[VisualArtifact, ...]:
        self.calls.append(data.label)
        return ()


# ── Branching rule stubs ─────────────────────────────────────────────────────


class AlwaysMatchRule(IBranchingRule):
    def matches(self, event: Event) -> bool:
        return True

    def build_context(self, event: Event) -> PipelineContext:
        return PipelineContext(
            frame_extractor=StubFrameExtractor(num_frames=2),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer()],
        )


class SelectiveRule(IBranchingRule):
    def __init__(self, target_event_type: str):
        self._target = target_event_type

    def matches(self, event: Event) -> bool:
        return event.event_type == self._target

    def build_context(self, event: Event) -> PipelineContext:
        return PipelineContext(
            frame_extractor=StubFrameExtractor(num_frames=2),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer()],
        )


class NeverMatchRule(IBranchingRule):
    def matches(self, event: Event) -> bool:
        return False

    def build_context(self, event: Event) -> PipelineContext:
        raise AssertionError("Should never be called")


# ── Fake interface implementations for unit tests ────────────────────────────


class FakePipelineRunner(IPipelineRunner):
    def __init__(
        self,
        monitor: FakePipelineMonitor | None = None,
        fail_submit_with: Exception | None = None,
    ) -> None:
        self._monitor = monitor
        self._fail_submit_with = fail_submit_with
        self.ran: list[tuple[str, Pipeline]] = []
        self.submitted: list[tuple[str, Pipeline]] = []
        self.cancelled: list[str] = []
        self.shutdown_calls: list[bool] = []

    def run(self, pipeline_id: str, pipeline: Pipeline) -> PipelineOutputs:
        self.ran.append((pipeline_id, pipeline))
        if self._monitor is not None:
            self._monitor.register(pipeline_id)
        try:
            return pipeline.run()
        finally:
            if self._monitor is not None:
                self._monitor.complete(pipeline_id)

    def submit(self, pipeline_id: str, pipeline: Pipeline) -> Future[PipelineOutputs]:
        if self._fail_submit_with is not None:
            raise self._fail_submit_with
        self.submitted.append((pipeline_id, pipeline))
        if self._monitor is not None:
            self._monitor.register(pipeline_id)
        return Future()

    def cancel(self, pipeline_id: str) -> bool:
        self.cancelled.append(pipeline_id)
        if self._monitor is not None:
            self._monitor.terminate(pipeline_id)
        return True

    def active_ids(self) -> list[str]:
        if self._monitor is None:
            return []
        return self._monitor.active_ids()

    def snapshot(self, pipeline_id: str) -> PipelineRunSnapshot | None:
        if self._monitor is None:
            return None
        return self._monitor.snapshot(pipeline_id)

    def snapshots(self) -> list[PipelineRunSnapshot]:
        if self._monitor is None:
            return []
        return self._monitor.snapshots()

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls.append(wait)


class FakePipelineMonitor(IPipelineMonitor):
    def __init__(self) -> None:
        self._active: set[str] = set()
        self.completed: list[str] = []
        self.terminated: list[str] = []

    def register(self, pipeline_id: str) -> None:
        self._active.add(pipeline_id)

    def mark_running(self, pipeline_id: str, attempt: int) -> None:
        self._active.add(pipeline_id)

    def complete(self, pipeline_id: str) -> None:
        self._active.discard(pipeline_id)
        self.completed.append(pipeline_id)

    def fail(self, pipeline_id: str, error: Exception | str, attempt: int) -> None:
        self._active.discard(pipeline_id)
        self.completed.append(pipeline_id)

    def terminate(self, pipeline_id: str) -> None:
        self._active.discard(pipeline_id)
        self.terminated.append(pipeline_id)

    def active_ids(self) -> list[str]:
        return list(self._active)

    def snapshot(self, pipeline_id: str) -> PipelineRunSnapshot | None:
        return None

    def snapshots(self) -> list[PipelineRunSnapshot]:
        return []


class FailingOncePipelineMonitor(InMemoryPipelineMonitor):
    def __init__(self) -> None:
        super().__init__()
        self._should_fail = True

    def register(self, pipeline_id: str) -> None:
        if self._should_fail:
            self._should_fail = False
            raise RuntimeError("register failed")
        super().register(pipeline_id)


class RecordingPipelineFactory(IPipelineFactory):
    def __init__(self) -> None:
        self.created: list[PipelineContext] = []

    def create(
        self,
        context: PipelineContext,
        event_bus: IEventBus | None = None,
        pipeline_id: str | None = None,
        execution_metadata: dict[str, Any] | None = None,
    ) -> Pipeline:
        self.created.append(context)
        return Pipeline(
            context,
            event_bus=event_bus,
            pipeline_id=pipeline_id,
            execution_metadata=execution_metadata,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_context(signal_extractor: ISignalExtractor | None = None) -> PipelineContext:
    return PipelineContext(
        frame_extractor=StubFrameExtractor(),
        signal_extractor=signal_extractor or StubSignalExtractor(),
        analyzers=[StubAnalyzer()],
    )


def _make_failing_context() -> PipelineContext:
    return PipelineContext(
        frame_extractor=StubFrameExtractor(),
        signal_extractor=StubSignalExtractor(),
        analyzers=[FailingAnalyzer()],
    )


def _make_pipeline_event(pipeline_id: str, context: PipelineContext) -> Event:
    return PipelineEvent.create(
        pipeline_id=pipeline_id,
        context=context,
        source="test",
    )


def _event_pipeline_id(event: Event) -> str:
    return event.require("pipeline_id")


def _event_result_count(event: Event) -> int:
    return event.require("result_count")


def _make_orchestrator_with_branching(
    rules: list[IBranchingRule],
    signal_extractor: ISignalExtractor | None = None,
) -> tuple[EventBus, PipelineOrchestrator, PipelineContext]:
    """Return (trigger_bus, orchestrator, primary_context) wired for branching."""
    trigger_bus = EventBus()
    domain_bus = EventBus()
    monitor = InMemoryPipelineMonitor()
    runner = ThreadedPipelineRunner(
        monitor=monitor,
        retry_policy=NoRetryPolicy(),
        lifecycle_bus=trigger_bus,
    )
    BranchingCoordinator(
        event_bus=domain_bus,
        rules=rules,
        trigger_bus=trigger_bus,
    )
    orchestrator = PipelineOrchestrator(
        runner=runner,
        bus=trigger_bus,
        domain_bus=domain_bus,
    )
    context = _make_context(signal_extractor or EmittingSignalExtractor())
    return trigger_bus, orchestrator, context


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

        received: list[Event] = []
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

    def test_pipeline_injects_pipeline_id_into_domain_events(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("test_event", received.append)
        context = _make_context(EmittingSignalExtractor())

        Pipeline(context, event_bus=bus, pipeline_id="pipeline-123").run()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].payload["pipeline_id"], "pipeline-123")


class BranchingRuleTests(unittest.TestCase):
    def test_always_match_rule(self):
        rule = AlwaysMatchRule()
        event = Event("any_event", "src")
        self.assertTrue(rule.matches(event))
        self.assertIsNotNone(rule.build_context(event).frame_extractor)

    def test_selective_rule_matches_target(self):
        rule = SelectiveRule("tracking_lost")
        self.assertTrue(rule.matches(Event("tracking_lost", "src")))
        self.assertFalse(rule.matches(Event("other_event", "src")))

    def test_never_match_rule(self):
        self.assertFalse(NeverMatchRule().matches(Event("any", "src")))


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
        domain_bus.dispatch(Event("any", "src"))

        self.assertEqual(dispatched, [])

    def test_dispatch_on_matching_rule(self):
        domain_bus = EventBus()
        trigger_bus = EventBus()
        dispatched: list[PipelineEvent] = []
        trigger_bus.subscribe(PipelineEvent.event_type, dispatched.append)

        BranchingCoordinator(event_bus=domain_bus, rules=[AlwaysMatchRule()], trigger_bus=trigger_bus)
        domain_bus.dispatch(Event("any", "src"))

        self.assertEqual(len(dispatched), 1)
        self.assertTrue(_event_pipeline_id(dispatched[0]).startswith("secondary-"))

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
        domain_bus.dispatch(Event("test_event", "src"))

        self.assertEqual(dispatched, [])


class PipelineOrchestratorUnitTests(unittest.TestCase):
    """Unit tests using fake interface implementations — no real threads."""

    def _make(
        self,
    ) -> tuple[
        EventBus,
        FakePipelineRunner,
        FakePipelineMonitor,
        PipelineOrchestrator,
    ]:
        bus = EventBus()
        monitor = FakePipelineMonitor()
        runner = FakePipelineRunner(monitor)
        orch = PipelineOrchestrator(runner=runner, bus=bus)
        return bus, runner, monitor, orch

    def test_pipeline_event_triggers_monitor_runner(self):
        bus, runner, monitor, orch = self._make()
        context = _make_context()
        bus.dispatch(_make_pipeline_event("x", context))

        self.assertEqual(len(runner.submitted), 1)
        self.assertEqual(runner.submitted[0][0], "x")
        self.assertIn("x", monitor._active)

    def test_run_executes_synchronously_without_event_bus(self):
        _, runner, _, orch = self._make()

        outputs = orch.run(_make_context())

        self.assertEqual(len(outputs.results), 1)
        self.assertEqual(len(runner.ran), 1)
        self.assertEqual(runner.submitted, [])

    def test_submit_executes_through_runner_without_event_bus(self):
        monitor = FakePipelineMonitor()
        runner = FakePipelineRunner(monitor)
        orch = PipelineOrchestrator(runner=runner)

        future = orch.submit(_make_context(), pipeline_id="direct")

        self.assertIsInstance(future, Future)
        self.assertEqual(len(runner.submitted), 1)
        self.assertIn("direct", monitor._active)

    def test_active_ids_delegates_to_runner(self):
        bus, runner, monitor, orch = self._make()
        monitor.register("p1")
        monitor.register("p2")
        self.assertCountEqual(orch.active_ids(), ["p1", "p2"])

    def test_default_runner_owns_internal_monitor(self):
        orch = PipelineOrchestrator()

        try:
            self.assertEqual(orch.active_ids(), [])
        finally:
            orch.shutdown()

    def test_shutdown_delegates_to_runner(self):
        _, runner, _, orch = self._make()

        orch.shutdown(wait=False)

        self.assertEqual(runner.shutdown_calls, [False])

    def test_pipeline_factory_is_used_to_create_pipeline(self):
        runner = FakePipelineRunner()
        factory = RecordingPipelineFactory()
        orch = PipelineOrchestrator(runner=runner, pipeline_factory=factory)
        context = _make_context()

        orch.submit(context, pipeline_id="factory")

        self.assertEqual(factory.created, [context])
        self.assertEqual(len(runner.submitted), 1)

    def test_invalid_pipeline_event_is_ignored(self):
        bus = EventBus()
        runner = FakePipelineRunner()
        PipelineOrchestrator(runner=runner, bus=bus)

        bus.dispatch(
            Event(
                event_type=PipelineEvent.event_type,
                source="test",
                payload={"pipeline_id": "invalid", "context": object()},
            )
        )

        self.assertEqual(runner.submitted, [])

    def test_duplicate_pipeline_event_is_ignored(self):
        bus = EventBus()
        runner = FakePipelineRunner(
            fail_submit_with=PipelineRunAlreadyActiveError("already active")
        )
        PipelineOrchestrator(runner=runner, bus=bus)

        bus.dispatch(_make_pipeline_event("duplicate", _make_context()))

        self.assertEqual(runner.submitted, [])

    def test_terminate_delegates_best_effort_cancel_to_runner(self):
        bus, runner, monitor, orch = self._make()
        context = _make_context()
        bus.dispatch(_make_pipeline_event("p1", context))

        cancelled = orch.terminate("p1")

        self.assertTrue(cancelled)
        self.assertIn("p1", runner.cancelled)
        self.assertIn("p1", monitor.terminated)

    def test_multiple_events_each_trigger_chain(self):
        bus, runner, monitor, orch = self._make()
        for i in range(3):
            bus.dispatch(_make_pipeline_event(f"p{i}", _make_context()))

        self.assertEqual(len(runner.submitted), 3)


class PipelineOrchestratorIntegrationTests(unittest.TestCase):
    """Integration tests with real threads, real pipelines."""

    def test_pipeline_runs_and_emits_after_run(self):
        bus = EventBus()
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, lifecycle_bus=bus)
        orch = PipelineOrchestrator(runner=runner, bus=bus)

        payloads: list[Event] = []
        bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)

        context = _make_context()
        bus.dispatch(_make_pipeline_event("run-1", context))

        _wait_until_idle(orch)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(_event_pipeline_id(payloads[0]), "run-1")
        self.assertEqual(_event_result_count(payloads[0]), 1)

    def test_before_run_fires_before_after_run(self):
        bus = EventBus()
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, lifecycle_bus=bus)
        orch = PipelineOrchestrator(runner=runner, bus=bus)

        order: list[str] = []
        bus.subscribe(PipelineLifecycleEvent.BEFORE_RUN, lambda _: order.append("before"))
        bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, lambda _: order.append("after"))

        bus.dispatch(_make_pipeline_event("x", _make_context()))
        _wait_until_idle(orch)

        self.assertEqual(order, ["before", "after"])

    def test_no_active_ids_after_completion(self):
        bus = EventBus()
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, lifecycle_bus=bus)
        orch = PipelineOrchestrator(runner=runner, bus=bus)

        bus.dispatch(_make_pipeline_event("x", _make_context()))
        _wait_until_idle(orch)

        self.assertEqual(orch.active_ids(), [])

    def test_submit_returns_future_with_results(self):
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor)

        future = runner.submit("async", Pipeline(_make_context()))
        outputs = future.result(timeout=5)
        snapshot = runner.snapshot("async")

        self.assertEqual(len(outputs.results), 1)
        self.assertEqual(monitor.active_ids(), [])
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.state, PipelineRunState.SUCCEEDED)
        self.assertEqual(snapshot.attempt, 1)
        self.assertIsNone(snapshot.error)
        self.assertIsNotNone(snapshot.submitted_at)
        self.assertIsNotNone(snapshot.started_at)
        self.assertIsNotNone(snapshot.completed_at)

    def test_cancelled_pipeline_emits_lifecycle_event(self):
        bus = EventBus()
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, lifecycle_bus=bus, max_workers=1)
        started = ThreadingEvent()
        release = ThreadingEvent()
        cancelled: list[Event] = []
        bus.subscribe(PipelineLifecycleEvent.CANCELLED, cancelled.append)
        blocking_context = PipelineContext(
            frame_extractor=BlockingFrameExtractor(started, release),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer()],
        )

        running_future = runner.submit("running", Pipeline(blocking_context))
        self.assertTrue(started.wait(timeout=2))
        runner.submit("queued", Pipeline(_make_context()))

        try:
            self.assertTrue(runner.cancel("queued"))
            self.assertEqual(len(cancelled), 1)
            self.assertEqual(_event_pipeline_id(cancelled[0]), "queued")
        finally:
            release.set()

        running_future.result(timeout=5)

    def test_duplicate_pipeline_id_emits_rejected_lifecycle_event(self):
        bus = EventBus()
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, lifecycle_bus=bus, max_workers=1)
        started = ThreadingEvent()
        release = ThreadingEvent()
        rejected: list[Event] = []
        bus.subscribe(PipelineLifecycleEvent.REJECTED, rejected.append)
        blocking_context = PipelineContext(
            frame_extractor=BlockingFrameExtractor(started, release),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer()],
        )

        future = runner.submit("same", Pipeline(blocking_context))
        self.assertTrue(started.wait(timeout=2))

        try:
            with self.assertRaises(PipelineRunAlreadyActiveError):
                runner.run("same", Pipeline(_make_context()))
            self.assertEqual(len(rejected), 1)
            self.assertEqual(_event_pipeline_id(rejected[0]), "same")
        finally:
            release.set()

        future.result(timeout=5)

    def test_submit_after_shutdown_emits_submit_failed_lifecycle_event(self):
        bus = EventBus()
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, lifecycle_bus=bus)
        submit_failed: list[Event] = []
        bus.subscribe(PipelineLifecycleEvent.SUBMIT_FAILED, submit_failed.append)
        runner.shutdown()

        with self.assertRaises(RuntimeError):
            runner.submit("closed", Pipeline(_make_context()))

        self.assertEqual(len(submit_failed), 1)
        self.assertEqual(_event_pipeline_id(submit_failed[0]), "closed")

    def test_duplicate_pipeline_id_rejected_across_submit_and_run(self):
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, max_workers=1)
        started = ThreadingEvent()
        release = ThreadingEvent()
        blocking_context = PipelineContext(
            frame_extractor=BlockingFrameExtractor(started, release),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer()],
        )

        future = runner.submit("same", Pipeline(blocking_context))
        self.assertTrue(started.wait(timeout=2))

        try:
            with self.assertRaisesRegex(PipelineRunAlreadyActiveError, "already running"):
                runner.run("same", Pipeline(_make_context()))
        finally:
            release.set()

        future.result(timeout=5)
        self.assertEqual(monitor.active_ids(), [])

    def test_cancel_is_best_effort_for_queued_async_work(self):
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, max_workers=1)
        started = ThreadingEvent()
        release = ThreadingEvent()
        blocking_context = PipelineContext(
            frame_extractor=BlockingFrameExtractor(started, release),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer()],
        )

        running_future = runner.submit("running", Pipeline(blocking_context))
        self.assertTrue(started.wait(timeout=2))
        queued_future = runner.submit("queued", Pipeline(_make_context()))

        try:
            self.assertTrue(runner.cancel("queued"))
            self.assertTrue(queued_future.cancelled())
            self.assertFalse(runner.cancel("running"))
            queued_snapshot = runner.snapshot("queued")
            self.assertIsNotNone(queued_snapshot)
            self.assertEqual(queued_snapshot.state, PipelineRunState.CANCELLED)
        finally:
            release.set()

        running_future.result(timeout=5)
        self.assertEqual(monitor.active_ids(), [])

    def test_failed_pipeline_records_failed_snapshot(self):
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor)

        future = runner.submit("failing", Pipeline(_make_failing_context()))

        with self.assertRaises(PipelineExecutionError):
            future.result(timeout=5)

        snapshot = runner.snapshot("failing")
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.state, PipelineRunState.FAILED)
        self.assertEqual(snapshot.attempt, 1)
        self.assertIn("analysis failed", snapshot.error or "")
        self.assertEqual(monitor.active_ids(), [])

    def test_submit_rolls_back_active_id_when_monitor_registration_fails(self):
        monitor = FailingOncePipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor)

        with self.assertRaisesRegex(RuntimeError, "register failed"):
            runner.submit("retryable", Pipeline(_make_context()))

        future = runner.submit("retryable", Pipeline(_make_context()))
        outputs = future.result(timeout=5)

        self.assertEqual(len(outputs.results), 1)
        self.assertEqual(monitor.active_ids(), [])


class PipelineVisualizationTests(unittest.TestCase):
    def test_default_visualizers_receive_all_analyzer_results(self):
        visualizer = RecordingVisualizer()
        context = PipelineContext(
            frame_extractor=StubFrameExtractor(),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer(), OtherAnalyzer()],
            visualizers=[visualizer],
        )

        Pipeline(context).run()

        self.assertEqual(visualizer.calls, ["stub", "other"])

    def test_visualizer_binding_targets_selected_results_only(self):
        visualizer = RecordingVisualizer()
        context = PipelineContext(
            frame_extractor=StubFrameExtractor(),
            signal_extractor=StubSignalExtractor(),
            analyzers=[StubAnalyzer(), OtherAnalyzer()],
            visualizer_bindings=[
                VisualizerBinding(visualizer=visualizer, result_indices=(1,)),
            ],
        )

        Pipeline(context).run()

        self.assertEqual(visualizer.calls, ["other"])

    def test_fluent_builder_can_bind_visualizer_to_selected_results(self):
        visualizer = RecordingVisualizer()
        context = (
            FluentPipelineBuilder()
            .with_frame_extractor(StubFrameExtractor())
            .with_signal_extractor(StubSignalExtractor())
            .with_analyzers([StubAnalyzer(), OtherAnalyzer()])
            .add_visualizer_for_results(visualizer, [0])
            .build_context()
        )

        Pipeline(context).run()

        self.assertEqual(visualizer.calls, ["stub"])


class OrchestratorBranchingTests(unittest.TestCase):
    def test_auto_spawn_on_domain_event(self):
        trigger_bus, orch, context = _make_orchestrator_with_branching(rules=[AlwaysMatchRule()])
        payloads: list[Event] = []
        trigger_bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)
        orch.submit(context, pipeline_id="primary")

        _wait_until_idle(orch, timeout=10)

        pipeline_ids = {_event_pipeline_id(p) for p in payloads}
        self.assertIn("primary", pipeline_ids)
        secondary_ids = {pid for pid in pipeline_ids if pid.startswith("secondary-")}
        self.assertEqual(len(secondary_ids), 1)

    def test_selective_rule_ignores_non_matching(self):
        trigger_bus, orch, context = _make_orchestrator_with_branching(rules=[SelectiveRule("other_event")])
        payloads: list[Event] = []
        trigger_bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)
        orch.submit(context, pipeline_id="primary")

        _wait_until_idle(orch, timeout=10)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(_event_pipeline_id(payloads[0]), "primary")

    def test_multiple_rules_spawn_multiple_secondaries(self):
        trigger_bus, orch, context = _make_orchestrator_with_branching(rules=[AlwaysMatchRule(), AlwaysMatchRule()])
        payloads: list[Event] = []
        trigger_bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)
        orch.submit(context, pipeline_id="primary")

        _wait_until_idle(orch, timeout=10)

        secondary_ids = {_event_pipeline_id(p) for p in payloads if _event_pipeline_id(p).startswith("secondary-")}
        self.assertEqual(len(secondary_ids), 2)

    def test_domain_events_do_not_reach_lifecycle_handlers(self):
        trigger_bus, orch, context = _make_orchestrator_with_branching(rules=[AlwaysMatchRule()])
        lifecycle_payloads: list = []
        trigger_bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, lifecycle_payloads.append)
        orch.submit(context, pipeline_id="primary")

        _wait_until_idle(orch, timeout=10)

        for p in lifecycle_payloads:
            self.assertIsInstance(p, Event)


class FluentBuilderContextTests(unittest.TestCase):
    def test_fluent_context_can_run_with_orchestrator_branching(self):
        trigger_bus = EventBus()
        domain_bus = EventBus()
        monitor = InMemoryPipelineMonitor()
        runner = ThreadedPipelineRunner(monitor=monitor, lifecycle_bus=trigger_bus)
        BranchingCoordinator(
            event_bus=domain_bus,
            rules=[AlwaysMatchRule()],
            trigger_bus=trigger_bus,
        )
        orchestrator = PipelineOrchestrator(
            runner=runner,
            bus=trigger_bus,
            domain_bus=domain_bus,
        )
        payloads: list[Event] = []
        trigger_bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)

        context = (
            FluentPipelineBuilder()
            .with_frame_extractor(StubFrameExtractor())
            .with_signal_extractor(EmittingSignalExtractor())
            .add_analyzer(StubAnalyzer())
            .build_context()
        )
        orchestrator.submit(context, pipeline_id="primary")

        _wait_until_idle(orchestrator, timeout=10)

        pipeline_ids = {_event_pipeline_id(p) for p in payloads}
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

    def test_fluent_context_can_run_without_event_bus(self):
        context = _make_context()
        orchestrator = PipelineOrchestrator()

        outputs = orchestrator.run(context)

        self.assertEqual(len(outputs.results), 1)

    def test_fluent_context_can_submit_with_event_bus(self):
        bus = EventBus()
        payloads: list[Event] = []
        bus.subscribe(PipelineLifecycleEvent.AFTER_RUN, payloads.append)
        orchestrator = PipelineOrchestrator(bus=bus)

        context = _make_context()
        orchestrator.submit(context, pipeline_id="solo")

        _wait_until_idle(orchestrator, timeout=10)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(_event_pipeline_id(payloads[0]), "solo")


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
