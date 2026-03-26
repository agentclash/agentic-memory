from __future__ import annotations

import sys
from typing import TextIO

from events.bus import EventBus, MemoryEvent


class ConsoleLogger:
    """Human-readable console subscriber for memory lifecycle events."""

    def __init__(self, stream: TextIO | None = None):
        self._stream = stream or sys.stderr

    def register(self, bus: EventBus) -> None:
        bus.subscribe("memory.stored", self.on_memory_stored)
        bus.subscribe("memory.retrieved", self.on_memory_retrieved)
        bus.subscribe("memory.ranked", self.on_memory_ranked)
        bus.subscribe("memory.accessed", self.on_memory_accessed)

    def on_memory_stored(self, event: MemoryEvent) -> None:
        content = self._truncate(str(event.data.get("content", "")))
        print(
            f"[EVENT] {event.event_type} | record_id={event.data.get('record_id')} "
            f"| type={event.data.get('memory_type')} | content={content}",
            file=self._stream,
        )

    def on_memory_retrieved(self, event: MemoryEvent) -> None:
        top_similarity = event.data.get("top_similarity")
        top_display = "n/a" if top_similarity is None else f"{float(top_similarity):.4f}"
        print(
            f"[EVENT] {event.event_type} | query={event.data.get('query')!r} "
            f"| candidates={event.data.get('candidate_count', 0)} | top_sim={top_display}",
            file=self._stream,
        )

    def on_memory_ranked(self, event: MemoryEvent) -> None:
        results = event.data.get("results", ())
        top_score = "n/a"
        if results:
            top_score = f"{float(results[0]['final_score']):.4f}"
        print(
            f"[EVENT] {event.event_type} | query={event.data.get('query')!r} "
            f"| top_score={top_score} | results={len(results)}",
            file=self._stream,
        )

    def on_memory_accessed(self, event: MemoryEvent) -> None:
        print(
            f"[EVENT] {event.event_type} | record_id={event.data.get('record_id')} "
            f"| count={event.data.get('access_count')}",
            file=self._stream,
        )

    @staticmethod
    def _truncate(text: str, limit: int = 72) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
