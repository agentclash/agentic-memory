from abc import ABC, abstractmethod

from models.base import MemoryRecord


class BaseStore(ABC):
    """All memory stores implement this interface — semantic, episodic, procedural."""

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
    def update_access(self, record_id: str) -> None:
        """Bump access_count and last_accessed_at for a retrieved record."""
        ...
