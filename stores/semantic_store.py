import chromadb
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

import config
from events.bus import EventBus
from models.semantic import SemanticMemory
from stores.base import BaseStore
from stores.media_store import MediaStore
from utils.embeddings import GeminiEmbedder, TextEmbedder


class SemanticStore(BaseStore):
    """ChromaDB-backed store for semantic (factual) memories."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        embedder: TextEmbedder | None = None,
        media_store: MediaStore | None = None,
    ):
        super().__init__(event_bus=event_bus)
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self._collection = client.get_or_create_collection(
            name="semantic_memories",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder or GeminiEmbedder()
        self._media_store = media_store

    # ── write ──────────────────────────────────────────────────────────────

    def store(self, record: SemanticMemory) -> str:
        owned_media_ref = None
        try:
            owned_media_ref = self._ensure_owned_media(record)
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
                    "has_media": record.has_media,
                    "importance": record.importance,
                    **({"media_ref": record.media_ref} if record.media_ref else {}),
                },
            )
            return record.id
        except Exception:
            if owned_media_ref and self._media_store is not None:
                self._media_store.delete(owned_media_ref)
                if record.media_ref == owned_media_ref:
                    record.media_ref = None
            raise

    # ── read ───────────────────────────────────────────────────────────────

    def get_by_id(self, record_id: str) -> SemanticMemory | None:
        result = self._collection.get(
            ids=[record_id],
            include=["embeddings", "documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return self._from_result(result, 0)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[SemanticMemory, float]]:
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
            # ChromaDB cosine distance = 1 - similarity
            similarity = 1.0 - result["distances"][0][i]
            pairs.append((record, similarity))
        return pairs

    # ── serialisation helpers ──────────────────────────────────────────────
    # ChromaDB metadata must be flat (str/int/float/bool only — no nested dicts).

    def update_access(self, record_id: str) -> None:
        now = datetime.now(timezone.utc)
        result = self._collection.get(ids=[record_id], include=["metadatas"])
        if not result["ids"]:
            return
        meta = result["metadatas"][0]
        meta["access_count"] = int(meta.get("access_count", 0)) + 1
        meta["last_accessed_at"] = now.isoformat()
        self._collection.update(ids=[record_id], metadatas=[meta])

    # ── serialisation helpers ──────────────────────────────────────────────
    # ChromaDB metadata must be flat (str/int/float/bool only — no nested dicts).

    def _to_metadata(self, record: SemanticMemory) -> dict:
        return {
            "memory_type": record.memory_type,
            "modality": record.modality,
            "created_at": record.created_at.isoformat(),
            "last_accessed_at": record.last_accessed_at.isoformat() if record.last_accessed_at else "",
            "access_count": record.access_count,
            "importance": record.importance,
            "source": record.source or "",
            "category": record.category,
            "domain": record.domain or "",
            "confidence": record.confidence,
            "supersedes": record.supersedes or "",
            "related_ids_json": json.dumps(record.related_ids),
            "has_visual": record.has_visual,
            "media_ref": record.media_ref or "",
            "media_type": record.media_type or "",
            "text_description": record.text_description or "",
        }

    def _build_record(self, doc: str, id: str, embedding, meta: dict) -> SemanticMemory:
        last_accessed = meta.get("last_accessed_at", "")
        return SemanticMemory(
            content=doc,
            id=id,
            embedding=embedding,
            created_at=datetime.fromisoformat(meta["created_at"]),
            last_accessed_at=datetime.fromisoformat(last_accessed) if last_accessed else None,
            access_count=int(meta.get("access_count", 0)),
            importance=float(meta["importance"]),
            source=meta["source"] or None,
            category=meta["category"],
            domain=meta.get("domain") or None,
            confidence=float(meta["confidence"]),
            supersedes=meta.get("supersedes") or None,
            related_ids=json.loads(meta.get("related_ids_json", "[]")),
            has_visual=bool(meta.get("has_visual", False)),
            modality=meta["modality"],
            media_ref=meta.get("media_ref") or None,
            media_type=meta.get("media_type") or None,
            text_description=meta.get("text_description") or None,
        )

    def _embed_record(self, record: SemanticMemory) -> list[float]:
        if record.modality == "text":
            return self._embedder.embed_text(record.content)

        media_path = self._require_media_path(record)
        text_context = self._text_context(record)
        media_type = self._resolve_multimodal_media_type(record, media_path)

        # Semantic media records always carry meaningful text context in `content`,
        # so aggregate text + media into one vector rather than treating the media
        # as an isolated payload. This is intentionally asymmetric with the
        # episodic store, which keeps its modality-specific embedding strategy.
        return self._embed_multimodal(
            media_type,
            media_path,
            text_context,
            mime_type=self._resolve_mime_type(media_type, media_path),
        )

    def _embed_multimodal(
        self,
        media_type: str,
        media_path: Path,
        text_context: str,
        *,
        mime_type: str | None = None,
    ) -> list[float]:
        kwargs = {"text": text_context or None}
        kwargs[media_type] = str(media_path)
        kwargs[f"{media_type}_mime_type"] = mime_type
        return self._embedder.embed_multimodal(**kwargs)

    def _ensure_owned_media(self, record: SemanticMemory) -> str | None:
        if not record.media_ref or self._media_store is None:
            return None
        record.media_ref, copied = self._media_store.ensure_owned(record.media_ref, record.id)
        return record.media_ref if copied else None

    def _require_media_path(self, record: SemanticMemory) -> Path:
        if not record.media_ref:
            raise ValueError(f"Semantic {record.modality} memory requires media_ref")
        media_path = Path(record.media_ref)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {record.media_ref}")
        return media_path

    def _resolve_multimodal_media_type(self, record: SemanticMemory, media_path: Path) -> str:
        return MediaStore.resolve_media_type(media_path, record.media_type)

    def _resolve_mime_type(self, media_type: str, media_path: Path) -> str | None:
        guessed = mimetypes.guess_type(media_path.name)[0]
        if guessed:
            return guessed

        defaults = {
            "image": "image/png",
            "audio": "audio/mpeg",
            "video": "video/mp4",
            "pdf": "application/pdf",
        }
        return defaults.get(media_type)

    def _text_context(self, record: SemanticMemory) -> str:
        parts = [record.content]
        if record.text_description:
            parts.append(record.text_description)
        return "\n".join(part for part in parts if part)

    def _from_result(self, result: dict, index: int) -> SemanticMemory:
        """Deserialise from .get() result (flat lists)."""
        return self._build_record(
            result["documents"][index], result["ids"][index],
            result["embeddings"][index], result["metadatas"][index],
        )

    def _from_query_result(self, result: dict, index: int) -> SemanticMemory:
        """Deserialise from .query() result (nested lists — one list per query)."""
        return self._build_record(
            result["documents"][0][index], result["ids"][0][index],
            result["embeddings"][0][index], result["metadatas"][0][index],
        )
