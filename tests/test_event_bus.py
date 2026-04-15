"""Tests for Event and EventBus."""

from __future__ import annotations

import threading
import unittest

from library.core.events.Event import Event
from library.core.events.EventBus import EventBus


class EventTests(unittest.TestCase):
    """Verify Event creation and immutability."""

    def test_creation_with_defaults(self):
        event = Event(event_type="tracking_lost", source="MySE")
        self.assertEqual(event.event_type, "tracking_lost")
        self.assertEqual(event.source, "MySE")
        self.assertEqual(event.payload, {})
        self.assertIsInstance(event.timestamp, float)

    def test_creation_with_payload(self):
        event = Event(
            event_type="object_detected",
            source="Detector",
            payload={"frame_index": 42, "confidence": 0.95},
        )
        self.assertEqual(event.payload["frame_index"], 42)
        self.assertAlmostEqual(event.payload["confidence"], 0.95)

    def test_frozen_immutability(self):
        event = Event(event_type="test", source="src")
        with self.assertRaises(AttributeError):
            event.event_type = "other"  # type: ignore[misc]

    def test_repr_is_readable(self):
        event = Event("test_event", "source", {"key": "val"})
        r = repr(event)
        self.assertIn("test_event", r)
        self.assertIn("source", r)
        self.assertIn("key", r)


class EventBusPublishSubscribeTests(unittest.TestCase):
    """Core pub/sub functionality."""

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("evt_a", received.append)

        bus.dispatch(Event("evt_a", "src"))

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].event_type, "evt_a")

    def test_multiple_handlers_same_event(self):
        bus = EventBus()
        calls: list[str] = []
        bus.subscribe("evt", lambda e: calls.append("h1"))
        bus.subscribe("evt", lambda e: calls.append("h2"))

        bus.dispatch(Event("evt", "src"))

        self.assertEqual(calls, ["h1", "h2"])

    def test_handler_not_called_for_different_event(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("evt_a", received.append)

        bus.dispatch(Event("evt_b", "src"))

        self.assertEqual(len(received), 0)

    def test_unsubscribe(self):
        bus = EventBus()
        received: list[Event] = []
        handler = received.append
        bus.subscribe("evt", handler)

        bus.dispatch(Event("evt", "src"))
        self.assertEqual(len(received), 1)

        bus.unsubscribe("evt", handler)
        bus.dispatch(Event("evt", "src"))
        self.assertEqual(len(received), 1)  # unchanged

    def test_unsubscribe_nonexistent_handler_is_noop(self):
        bus = EventBus()
        bus.unsubscribe("evt", lambda e: None)  # should not raise


class EventBusWildcardTests(unittest.TestCase):
    """Wildcard handler tests."""

    def test_wildcard_subscription_receives_every_event(self):
        bus = EventBus()
        received: list[str] = []
        bus.subscribe(EventBus.WILDCARD, lambda e: received.append(e.event_type))

        bus.dispatch(Event("evt_a", "src"))
        bus.dispatch(Event("evt_b", "src"))
        bus.dispatch(Event("evt_c", "src"))

        self.assertEqual(received, ["evt_a", "evt_b", "evt_c"])

    def test_unsubscribe_removes_wildcard(self):
        bus = EventBus()
        received: list[Event] = []
        handler = received.append
        bus.subscribe(EventBus.WILDCARD, handler)

        bus.dispatch(Event("evt", "src"))
        self.assertEqual(len(received), 1)

        bus.unsubscribe(EventBus.WILDCARD, handler)
        bus.dispatch(Event("evt", "src"))
        self.assertEqual(len(received), 1)  # unchanged

    def test_specific_and_wildcard_both_fire(self):
        bus = EventBus()
        calls: list[str] = []
        bus.subscribe("evt", lambda e: calls.append("specific"))
        bus.subscribe(EventBus.WILDCARD, lambda e: calls.append("wildcard"))

        bus.dispatch(Event("evt", "src"))

        self.assertEqual(calls, ["specific", "wildcard"])


class EventBusErrorIsolationTests(unittest.TestCase):
    """Handler exceptions must not crash publish."""

    def test_handler_exception_is_isolated(self):
        bus = EventBus()
        calls: list[str] = []

        def bad_handler(e: Event) -> None:
            raise RuntimeError("boom")

        bus.subscribe("evt", bad_handler)
        bus.subscribe("evt", lambda e: calls.append("ok"))

        bus.dispatch(Event("evt", "src"))

        # Second handler still ran despite first raising
        self.assertEqual(calls, ["ok"])


class EventBusThreadSafetyTests(unittest.TestCase):
    """Verify concurrent publish/subscribe does not corrupt state."""

    def test_concurrent_publish(self):
        bus = EventBus()
        received: list[Event] = []
        lock = threading.Lock()

        def safe_append(e: Event) -> None:
            with lock:
                received.append(e)

        bus.subscribe("evt", safe_append)

        threads = [threading.Thread(target=lambda i=i: bus.dispatch(Event("evt", "src", {"i": i}))) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(received), 50)


class EventBusManagementTests(unittest.TestCase):
    """Tests for clear() and has_subscribers()."""

    def test_clear_removes_all_handlers(self):
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe("evt", received.append)
        bus.subscribe(EventBus.WILDCARD, received.append)

        bus.clear()
        bus.dispatch(Event("evt", "src"))

        self.assertEqual(len(received), 0)

    def test_has_subscribers(self):
        bus = EventBus()
        self.assertFalse(bus.has_subscribers("evt"))

        bus.subscribe("evt", lambda e: None)
        self.assertTrue(bus.has_subscribers("evt"))

    def test_has_subscribers_via_wildcard(self):
        bus = EventBus()
        self.assertFalse(bus.has_subscribers("any"))

        bus.subscribe(EventBus.WILDCARD, lambda e: None)
        self.assertTrue(bus.has_subscribers("any"))


if __name__ == "__main__":
    unittest.main()
