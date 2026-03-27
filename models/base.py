from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid

ALLOWED_MODALITIES = {"text", "image", "audio", "video", "multimodal"}
_LEGACY_MODALITY_ALIASES = {
    "pdf": "multimodal",
}


def normalize_modality(modality: str | None) -> str:
    if modality is None:
        resolved = "text"
    elif isinstance(modality, str):
        resolved = modality.strip().lower()
    else:
        raise ValueError("Unsupported modality type. Expected a string or null.")
    resolved = _LEGACY_MODALITY_ALIASES.get(resolved, resolved)
    if resolved not in ALLOWED_MODALITIES:
        supported = ", ".join(sorted(ALLOWED_MODALITIES))
        raise ValueError(f"Unsupported modality '{modality}'. Supported values: {supported}")
    return resolved


@dataclass(kw_only=True)
class MemoryRecord:
    content: str                                # required — no default
    memory_type: str                            # required — no default
    modality: str = "text"

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    importance: float = 0.5

    embedding: Optional[list[float]] = None
    embedding_dims: int = 768

    source: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    media_ref: Optional[str] = None
    media_type: Optional[str] = None
    text_description: Optional[str] = None

    def __post_init__(self) -> None:
        # Subclasses that override __post_init__ must call super().__post_init__()
        # so modality normalization remains part of the record contract.
        self.modality = normalize_modality(self.modality)

    @property
    def has_media(self) -> bool:
        return bool(self.media_ref)
