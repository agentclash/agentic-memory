import json
from datetime import datetime, timezone
from heapq import nlargest
from pathlib import Path

import chromadb

import config
from events.bus import EventBus
from models.episodic import EpisodicMemory
from stores.base import BaseStore
from utils.embeddings import GeminiEmbedder, TextEmbedder

_MEDIA_EMBED_METHODS = {
    "image": "embed_image",
    "audio": "embed_audio",
    "video": "embed_video",
    "pdf": "embed_pdf",
}

_DEFAULT_MIME_TYPES = {
    "image": "image/png",
    "audio": "audio/mpeg",
    "video": "video/mp4",
    "pdf": "application/pdf",
}


class EpisodicStoreError(RuntimeError):
    """Raised when episodic persistence fails before the record can be stored."""


class MediaTooLargeError(EpisodicStoreError):
    """Raised when a media-backed record exceeds the direct-embedding guardrail."""


class EpisodicStore(BaseStore):
    """ChromaDB-backed store for episodic memories."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        embedder: TextEmbedder | None = None,
        max_media_bytes: int | None = None,
    ):
        super().__init__(event_bus=event_bus)
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self._collection = client.get_or_create_collection(
            name="episodic_memories",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder or GeminiEmbedder()
        self._max_media_bytes = (
            config.MEDIA_EMBED_MAX_BYTES if max_media_bytes is None else max_media_bytes
        )

    def store(self, record: EpisodicMemory) -> str:
        embedding = self._embed_record(record)
        record.embedding = embedding

        self._collection.add(
            ids=[record.id],
            embeddings=[embedding],
            documents=[record.content],
            metadatas=[self._to_metadata(record)],
        )
        self._emit_event(
            "memory.stored",
            {
                "record_id": record.id,
                "memory_type": record.memory_type,
                "content": record.content,
                "modality": record.modality,
                "importance": record.importance,
                "session_id": record.session_id,
                **({"media_ref": record.media_ref} if record.media_ref else {}),
            },
        )
        return record.id

    def get_by_id(self, record_id: str) -> EpisodicMemory | None:
        result = self._collection.get(
            ids=[record_id],
            include=["embeddings", "documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return self._from_result(result, 0)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[EpisodicMemory, float]]:
        query_embedding = self._embedder.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["embeddings", "documents", "metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        pairs = []
        for i in range(len(result["ids"][0])):
            record = self._from_query_result(result, i)
            similarity = 1.0 - result["distances"][0][i]
            pairs.append((record, similarity))
        return pairs

    def get_by_session(self, session_id: str) -> list[EpisodicMemory]:
        """Return all episodic records for a session, ordered by turn then created_at."""
        records = self._all_records(
            where={"session_id": session_id},
            include_embeddings=False,
        )
        return sorted(records, key=self._session_sort_key)

    def get_recent(self, n: int) -> list[EpisodicMemory]:
        """Return the most recent episodic records across sessions, newest first."""
        if n <= 0:
            return []
        return nlargest(
            n,
            self._all_records(include_embeddings=False),
            key=lambda record: self._as_utc(record.created_at),
        )

    def get_by_time_range(
        self,
        start: datetime,
        end: datetime,
    ) -> list[EpisodicMemory]:
        """Return records inside an inclusive time window.

        Naive datetimes are treated as UTC so tests and callers have stable behavior.
        Returned records are sorted chronologically by created_at.
        """
        start_utc = self._as_utc(start)
        end_utc = self._as_utc(end)
        if start_utc > end_utc:
            raise ValueError("start must be earlier than or equal to end")

        records = [
            record
            for record in self._all_records(include_embeddings=False)
            if start_utc <= self._as_utc(record.created_at) <= end_utc
        ]
        return sorted(records, key=lambda record: self._as_utc(record.created_at))

    def update_access(self, record_id: str) -> None:
        now = datetime.now(timezone.utc)
        result = self._collection.get(ids=[record_id], include=["metadatas"])
        if not result["ids"]:
            return
        meta = result["metadatas"][0]
        meta["access_count"] = int(meta.get("access_count", 0)) + 1
        meta["last_accessed_at"] = now.isoformat()
        self._collection.update(ids=[record_id], metadatas=[meta])

    def _embed_record(self, record: EpisodicMemory) -> list[float]:
        if record.modality == "text" or not record.media_ref:
            return self._embedder.embed_text(self._fallback_text(record))

        embed_method_name = _MEDIA_EMBED_METHODS.get(record.modality)
        embed_method = getattr(self._embedder, embed_method_name, None) if embed_method_name else None
        if embed_method is None:
            return self._embedder.embed_text(self._fallback_text(record))

        media_path = Path(record.media_ref)
        if not media_path.exists():
            raise EpisodicStoreError(
                f"Cannot store episodic media record: file not found at {record.media_ref}"
            )

        size_bytes = media_path.stat().st_size
        if size_bytes > self._max_media_bytes:
            raise MediaTooLargeError(
                "Cannot store episodic media record via direct embedding: "
                f"modality={record.modality} size_bytes={size_bytes} "
                f"limit_bytes={self._max_media_bytes} path={record.media_ref}"
            )

        media_bytes = media_path.read_bytes()
        mime_type = record.source_mime_type or _DEFAULT_MIME_TYPES.get(record.modality, "application/octet-stream")
        try:
            return embed_method(media_bytes, mime_type=mime_type)
        except Exception as exc:
            # Some provider/media combinations are rejected even though the file is valid.
            # Preserve the record by falling back to text embedding instead of failing the write.
            try:
                fallback = self._embedder.embed_text(self._fallback_text(record))
            except Exception as fallback_exc:
                raise EpisodicStoreError(
                    "Failed to embed episodic media record: "
                    f"modality={record.modality} path={record.media_ref} mime_type={mime_type}"
                ) from fallback_exc

            record.metadata = {
                **record.metadata,
                "embedding_strategy": "text_fallback",
                "media_embed_error": str(exc),
            }
            return fallback

    def _fallback_text(self, record: EpisodicMemory) -> str:
        parts = [record.content]
        if record.summary:
            parts.append(record.summary)
        if record.text_description:
            parts.append(record.text_description)
        if record.participants:
            parts.append(" ".join(record.participants))
        return "\n".join(part for part in parts if part)

    def _to_metadata(self, record: EpisodicMemory) -> dict:
        return {
            "memory_type": record.memory_type,
            "modality": record.modality,
            "created_at": record.created_at.isoformat(),
            "last_accessed_at": record.last_accessed_at.isoformat() if record.last_accessed_at else "",
            "access_count": record.access_count,
            "importance": record.importance,
            "source": record.source or "",
            "metadata_json": json.dumps(record.metadata, sort_keys=True),
            "session_id": record.session_id,
            "has_turn_number": record.turn_number is not None,
            "turn_number": record.turn_number if record.turn_number is not None else 0,
            "participants_json": json.dumps(record.participants),
            "summary": record.summary or "",
            "has_emotional_valence": record.emotional_valence is not None,
            "emotional_valence": record.emotional_valence if record.emotional_valence is not None else 0.0,
            "media_ref": record.media_ref or "",
            "media_type": record.media_type or "",
            "text_description": record.text_description or "",
            "source_mime_type": record.source_mime_type or "",
        }

    def _all_records(
        self,
        *,
        where: dict | None = None,
        include_embeddings: bool = True,
    ) -> list[EpisodicMemory]:
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.insert(0, "embeddings")

        result = self._collection.get(
            where=where,
            include=include,
        )
        if not result["ids"]:
            return []
        return [self._from_result(result, i) for i in range(len(result["ids"]))]

    def _session_sort_key(self, record: EpisodicMemory) -> tuple[bool, int, datetime]:
        turn_number = record.turn_number if record.turn_number is not None else 0
        return (
            record.turn_number is None,
            turn_number,
            self._as_utc(record.created_at),
        )

    def _as_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _build_record(self, doc: str, record_id: str, embedding, meta: dict) -> EpisodicMemory:
        last_accessed = meta.get("last_accessed_at", "")
        return EpisodicMemory(
            content=doc,
            id=record_id,
            embedding=embedding,
            created_at=datetime.fromisoformat(meta["created_at"]),
            last_accessed_at=datetime.fromisoformat(last_accessed) if last_accessed else None,
            access_count=int(meta.get("access_count", 0)),
            importance=float(meta["importance"]),
            source=meta.get("source") or None,
            metadata=json.loads(meta.get("metadata_json", "{}")),
            modality=meta.get("modality", "text"),
            session_id=meta.get("session_id", "default"),
            turn_number=int(meta["turn_number"]) if meta.get("has_turn_number") else None,
            participants=json.loads(meta.get("participants_json", "[]")),
            summary=meta.get("summary") or None,
            emotional_valence=(
                float(meta["emotional_valence"]) if meta.get("has_emotional_valence") else None
            ),
            media_ref=meta.get("media_ref") or None,
            media_type=meta.get("media_type") or None,
            text_description=meta.get("text_description") or None,
            source_mime_type=meta.get("source_mime_type") or None,
        )

    def _from_result(self, result: dict, index: int) -> EpisodicMemory:
        embeddings = result.get("embeddings")
        return self._build_record(
            result["documents"][index],
            result["ids"][index],
            embeddings[index] if embeddings is not None else None,
            result["metadatas"][index],
        )

    def _from_query_result(self, result: dict, index: int) -> EpisodicMemory:
        return self._build_record(
            result["documents"][0][index],
            result["ids"][0][index],
            result["embeddings"][0][index],
            result["metadatas"][0][index],
        )
