from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from events import EventBus
from events.bus import MemoryEvent
from forgetting.contradiction import ContradictionDetector
from forgetting.service import ForgettingService
from retrieval.retriever import UnifiedRetriever
from stores.episodic_store import EpisodicStore
from stores.media_store import MediaStore
from stores.procedural_store import ProceduralStore
from stores.semantic_store import SemanticStore
from utils.embeddings import GeminiEmbedder, TextEmbedder


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class EventRecorder:
    def __init__(self, bus: EventBus, *, max_events: int = 200):
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        for event_type in (
            "memory.stored",
            "memory.updated",
            "memory.retrieved",
            "memory.ranked",
            "memory.accessed",
            "memory.contradiction_flagged",
            "memory.supersession_resolved",
            "memory.faded",
            "memory.forgotten",
            "forgetting.cycle_completed",
            "forgetting.cycle_dry_run",
        ):
            bus.subscribe(event_type, self._record)

    def _record(self, event: MemoryEvent) -> None:
        self._events.appendleft(
            {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "data": _jsonable(dict(event.data)),
            }
        )

    def snapshot(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._events)[:limit]


@dataclass(slots=True)
class MemoryRuntime:
    media_store: MediaStore
    bus: EventBus
    embedder: TextEmbedder
    semantic_store: SemanticStore
    episodic_store: EpisodicStore
    procedural_store: ProceduralStore
    retriever: UnifiedRetriever
    contradiction_detector: ContradictionDetector
    forgetting_service: ForgettingService
    events: EventRecorder


def _embedder_dimensions(embedder: TextEmbedder) -> int | None:
    return embedder.dimensions


def build_runtime(
    *,
    chroma_path: str | None = None,
    media_root: str | Path | None = None,
    embedder: TextEmbedder | None = None,
    embedding_dimensions: int | None = None,
    max_events: int = 200,
) -> MemoryRuntime:
    resolved_media_root = (
        Path(media_root) if media_root is not None else Path(config.MEDIA_STORAGE_PATH)
    )
    embedder_dimensions = _embedder_dimensions(embedder) if embedder is not None else None
    if (
        embedding_dimensions is not None
        and embedder_dimensions is not None
        and embedder_dimensions != embedding_dimensions
    ):
        raise ValueError(
            "Provided embedder dimensions do not match embedding_dimensions: "
            f"embedder={embedder_dimensions} embedding_dimensions={embedding_dimensions}"
        )

    resolved_dimensions = (
        embedding_dimensions
        if embedding_dimensions is not None
        else (
            embedder_dimensions
            if embedder_dimensions is not None
            else config.EMBEDDING_DIMENSIONS
        )
    )

    bus = EventBus()
    active_embedder = embedder or GeminiEmbedder(dimensions=resolved_dimensions)
    media_store = MediaStore(resolved_media_root)
    semantic_store = SemanticStore(
        chroma_path=chroma_path,
        event_bus=bus,
        embedder=active_embedder,
        media_store=media_store,
    )
    episodic_store = EpisodicStore(
        chroma_path=chroma_path,
        event_bus=bus,
        embedder=active_embedder,
        media_store=media_store,
    )
    procedural_store = ProceduralStore(
        chroma_path=chroma_path,
        event_bus=bus,
        embedder=active_embedder,
        media_store=media_store,
    )
    retriever = UnifiedRetriever(
        stores={
            "semantic": semantic_store,
            "episodic": episodic_store,
            "procedural": procedural_store,
        },
        event_bus=bus,
        embedding_dimensions=resolved_dimensions,
    )
    contradiction_detector = ContradictionDetector(semantic_store, event_bus=bus)
    forgetting_service = ForgettingService(
        semantic_store=semantic_store,
        episodic_store=episodic_store,
        procedural_store=procedural_store,
        media_store=media_store,
        event_bus=bus,
        contradiction_detector=contradiction_detector,
    )
    events = EventRecorder(bus, max_events=max_events)
    return MemoryRuntime(
        media_store=media_store,
        bus=bus,
        embedder=active_embedder,
        semantic_store=semantic_store,
        episodic_store=episodic_store,
        procedural_store=procedural_store,
        retriever=retriever,
        contradiction_detector=contradiction_detector,
        forgetting_service=forgetting_service,
        events=events,
    )
