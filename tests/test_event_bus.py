"""Tests for EventBus and DomainEvent."""
from __future__ import annotations

import threading
import unittest

from library.core.events.DomainEvent import DomainEvent
from library.core.events.EventBus import EventBus


class DomainEventTests(unittest.TestCase):
    """Verify DomainEvent creation and immutability."""

    def test_creation_with_defaults(self):
        event = DomainEvent(event_type="tracking_lost", source="MySE")
        self.assertEqual(event.event_type, "tracking_lost")
        self.assertEqual(event.source, "MySE")
        self.assertEqual(event.payload, {})
        self.assertIsInstance(event.timestamp, float)

    def test_creation_with_payload(self):
        event = DomainEvent(
            event_type="object_detected",
            source="Detector",
            payload={"frame_index": 42, "confidence": 0.95},
        )
        self.assertEqual(event.payload["frame_index"], 42)
        self.assertAlmostEqual(event.payload["confidence"], 0.95)

    def test_frozen_immutability(self):
        event = DomainEvent(event_type="test", source="src")
        with self.assertRaises(AttributeError):
            event.event_type = "other"  # type: ignore[misc]

    def test_repr_is_readable(self):
        event = DomainEvent("test_event", "source", {"key": "val"})
        r = repr(event)
        self.assertIn("test_event", r)
        self.assertIn("source", r)
        self.assertIn("key", r)


class EventBusPublishSubscribeTests(unittest.TestCase):
    """Core pub/sub functionality."""

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received: list[DomainEvent] = []
        bus.subscribe("evt_a", received.append)

        bus.publish(DomainEvent("evt_a", "src"))

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].event_type, "evt_a")

    def test_multiple_handlers_same_event(self):
        bus = EventBus()
        calls: list[str] = []
        bus.subscribe("evt", lambda e: calls.append("h1"))
        bus.subscribe("evt", lambda e: calls.append("h2"))

        bus.publish(DomainEvent("evt", "src"))

        self.assertEqual(calls, ["h1", "h2"])

    def test_handler_not_called_for_different_event(self):
        bus = EventBus()
        received: list[DomainEvent] = []
        bus.subscribe("evt_a", received.append)

        bus.publish(DomainEvent("evt_b", "src"))

        self.assertEqual(len(received), 0)

    def test_unsubscribe(self):
        bus = EventBus()
        received: list[DomainEvent] = []
        handler = received.append
        bus.subscribe("evt", handler)

        bus.publish(DomainEvent("evt", "src"))
        self.assertEqual(len(received), 1)

        bus.unsubscribe("evt", handler)
        bus.publish(DomainEvent("evt", "src"))
        self.assertEqual(len(received), 1)  # unchanged

    def test_unsubscribe_nonexistent_handler_is_noop(self):
        bus = EventBus()
        bus.unsubscribe("evt", lambda e: None)  # should not raise


class EventBusWildcardTests(unittest.TestCase):
    """subscribe_all wildcard handler tests."""

    def test_subscribe_all_receives_every_event(self):
        bus = EventBus()
        received: list[str] = []
        bus.subscribe_all(lambda e: received.append(e.event_type))

        bus.publish(DomainEvent("evt_a", "src"))
        bus.publish(DomainEvent("evt_b", "src"))
        bus.publish(DomainEvent("evt_c", "src"))

        self.assertEqual(received, ["evt_a", "evt_b", "evt_c"])

    def test_unsubscribe_all_removes_wildcard(self):
        bus = EventBus()
        received: list[DomainEvent] = []
        handler = received.append
        bus.subscribe_all(handler)

        bus.publish(DomainEvent("evt", "src"))
        self.assertEqual(len(received), 1)

        bus.unsubscribe_all(handler)
        bus.publish(DomainEvent("evt", "src"))
        self.assertEqual(len(received), 1)  # unchanged

    def test_specific_and_wildcard_both_fire(self):
        bus = EventBus()
        calls: list[str] = []
        bus.subscribe("evt", lambda e: calls.append("specific"))
        bus.subscribe_all(lambda e: calls.append("wildcard"))

        bus.publish(DomainEvent("evt", "src"))

        self.assertEqual(calls, ["specific", "wildcard"])


class EventBusErrorIsolationTests(unittest.TestCase):
    """Handler exceptions must not crash publish."""

    def test_handler_exception_is_isolated(self):
        bus = EventBus()
        calls: list[str] = []

        def bad_handler(e: DomainEvent) -> None:
            raise RuntimeError("boom")

        bus.subscribe("evt", bad_handler)
        bus.subscribe("evt", lambda e: calls.append("ok"))

        bus.publish(DomainEvent("evt", "src"))

        # Second handler still ran despite first raising
        self.assertEqual(calls, ["ok"])


class EventBusThreadSafetyTests(unittest.TestCase):
    """Verify concurrent publish/subscribe does not corrupt state."""

    def test_concurrent_publish(self):
        bus = EventBus()
        received: list[DomainEvent] = []
        lock = threading.Lock()

        def safe_append(e: DomainEvent) -> None:
            with lock:
                received.append(e)

        bus.subscribe("evt", safe_append)

        threads = [
            threading.Thread(
                target=lambda i=i: bus.publish(
                    DomainEvent("evt", "src", {"i": i})
                )
            )
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(received), 50)


class EventBusManagementTests(unittest.TestCase):
    """Tests for clear() and has_subscribers()."""

    def test_clear_removes_all_handlers(self):
        bus = EventBus()
        received: list[DomainEvent] = []
        bus.subscribe("evt", received.append)
        bus.subscribe_all(received.append)

        bus.clear()
        bus.publish(DomainEvent("evt", "src"))

        self.assertEqual(len(received), 0)

    def test_has_subscribers(self):
        bus = EventBus()
        self.assertFalse(bus.has_subscribers("evt"))

        bus.subscribe("evt", lambda e: None)
        self.assertTrue(bus.has_subscribers("evt"))

    def test_has_subscribers_via_wildcard(self):
        bus = EventBus()
        self.assertFalse(bus.has_subscribers("any"))

        bus.subscribe_all(lambda e: None)
        self.assertTrue(bus.has_subscribers("any"))


if __name__ == "__main__":
    unittest.main()
