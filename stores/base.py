from abc import ABC, abstractmethod
from typing import Any

from events.bus import EventBus
from models.base import MemoryRecord


class BaseStore(ABC):
    """All memory stores implement this interface — semantic, episodic, procedural."""

    def __init__(self, event_bus: EventBus | None = None):
        self._event_bus = event_bus

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        if self._event_bus is not None:
            self._event_bus.emit(event_type, data)

    @abstractmethod
    def store(self, record: MemoryRecord) -> str:
        """Persist a record. Returns its id."""
        ...

    @abstractmethod
    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        """Fetch a single record by id. Returns None if not found."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[MemoryRecord, float]]:
        """Semantic search. Returns (record, similarity_score) pairs, highest first."""
        ...

    @abstractmethod
    def retrieve_by_vector(
        self,
        vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[MemoryRecord, float]]:
        """Vector search. Returns (record, similarity_score) pairs, highest first."""
        ...

    @abstractmethod
    def update_access(self, record_id: str) -> None:
        """Bump access_count and last_accessed_at for a retrieved record."""
        ...
