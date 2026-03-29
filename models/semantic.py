from dataclasses import dataclass, field

from models.base import MemoryRecord


@dataclass(kw_only=True)
class SemanticMemory(MemoryRecord):
    """Semantic memory with optional factual metadata and visual context hints."""

    memory_type: str = "semantic"
    category: str = "general"
    domain: str | None = None
    confidence: float = 1.0
    supersedes: str | None = None               # id of older record this replaces
    superseded_by: str | None = None            # id of newer canonical replacement
    related_ids: list[str] = field(default_factory=list)
    # Distinct from has_media: this can mark a fact as visually-oriented even
    # when no concrete image/file is attached to the record.
    has_visual: bool = False
