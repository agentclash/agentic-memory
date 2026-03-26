"""Verify the event bus snapshot and subscription behavior."""

import sys
import os
from datetime import timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events.bus import EventBus


def test_emit_notifies_subscriber():
    bus = EventBus()
    received = []

    def callback(event):
        received.append(event)

    bus.subscribe("memory.stored", callback)
    bus.emit("memory.stored", {"record_id": "abc"})

    assert len(received) == 1, f"Expected 1 event, got {len(received)}"
    assert received[0].event_type == "memory.stored"
    assert received[0].data["record_id"] == "abc"
    assert received[0].timestamp.tzinfo == timezone.utc
    print("  PASS  subscriber receives event with immutable snapshot")


def test_multiple_subscribers_receive_same_event():
    bus = EventBus()
    call_order = []

    def first(event):
        call_order.append(("first", event.event_type))

    def second(event):
        call_order.append(("second", event.event_type))

    bus.subscribe("memory.accessed", first)
    bus.subscribe("memory.accessed", second)
    bus.emit("memory.accessed", {"record_id": "xyz"})

    assert call_order == [
        ("first", "memory.accessed"),
        ("second", "memory.accessed"),
    ], f"Unexpected callback order: {call_order}"
    print("  PASS  multiple subscribers are notified")


def test_emit_without_subscribers_is_noop():
    bus = EventBus()
    event = bus.emit("memory.unknown", {"value": 1})

    assert event.event_type == "memory.unknown"
    assert event.data["value"] == 1
    print("  PASS  emit with no subscribers is a no-op")


def test_subscribers_only_receive_registered_events():
    bus = EventBus()
    received = []

    def callback(event):
        received.append(event.event_type)

    bus.subscribe("memory.stored", callback)
    bus.emit("memory.retrieved", {"query": "python"})
    bus.emit("memory.stored", {"record_id": "abc"})

    assert received == ["memory.stored"], f"Unexpected events received: {received}"
    print("  PASS  subscribers only receive registered event types")


def test_event_payload_is_snapshot_and_immutable():
    bus = EventBus()
    received = []
    payload = {"record_id": "abc", "results": [{"score": 0.9}]}

    def callback(event):
        received.append(event)

    bus.subscribe("memory.ranked", callback)
    bus.emit("memory.ranked", payload)

    payload["record_id"] = "mutated"
    payload["results"][0]["score"] = 0.1

    event = received[0]
    assert event.data["record_id"] == "abc"
    assert event.data["results"][0]["score"] == 0.9

    try:
        event.data["record_id"] = "nope"
        raise AssertionError("Expected immutable event data")
    except TypeError:
        pass

    print("  PASS  event payload is isolated from later mutations")


if __name__ == "__main__":
    print("Event bus tests:\n")
    test_emit_notifies_subscriber()
    test_multiple_subscribers_receive_same_event()
    test_emit_without_subscribers_is_noop()
    test_subscribers_only_receive_registered_events()
    test_event_payload_is_snapshot_and_immutable()
    print("\nAll tests passed.")
