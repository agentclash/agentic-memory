from dataclasses import dataclass, field
import uuid

from models.base import MemoryRecord


@dataclass(kw_only=True)
class EpisodicMemory(MemoryRecord):
    memory_type: str = "episodic"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_number: int | None = None
    participants: list[str] = field(default_factory=lambda: ["user", "agent"])
    summary: str | None = None
    emotional_valence: float | None = None
    emotional_profile: dict[str, float] = field(default_factory=dict)
    source_mime_type: str | None = None
