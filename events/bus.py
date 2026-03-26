from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Mapping


def _freeze(value: Any) -> Any:
    """Recursively freeze event payloads so subscribers see an immutable snapshot."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(val) for key, val in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    event_type: str
    data: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


EventCallback = Callable[[MemoryEvent], None]


class EventBus:
    """Synchronous pub/sub bus for memory lifecycle events."""

    def __init__(self):
        self._subscribers: dict[str, list[EventCallback]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        self._subscribers[event_type].append(callback)

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> MemoryEvent:
        event = MemoryEvent(
            event_type=event_type,
            data=_freeze(data or {}),
        )
        for callback in list(self._subscribers.get(event_type, [])):
            callback(event)
        return event
