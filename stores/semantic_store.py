import chromadb
from datetime import datetime, timezone

import config
from events.bus import EventBus
from models.semantic import SemanticMemory
from stores.base import BaseStore
from utils.embeddings import GeminiEmbedder, TextEmbedder


class SemanticStore(BaseStore):
    """ChromaDB-backed store for semantic (factual) memories."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        embedder: TextEmbedder | None = None,
    ):
        super().__init__(event_bus=event_bus)
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self._collection = client.get_or_create_collection(
            name="semantic_memories",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder or GeminiEmbedder()

    # ── write ──────────────────────────────────────────────────────────────

    def store(self, record: SemanticMemory) -> str:
        embedding = self._embedder.embed_text(record.content)
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
                **({"media_ref": record.media_ref} if record.media_ref else {}),
            },
        )
        return record.id

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
            "confidence": record.confidence,
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
            confidence=float(meta["confidence"]),
            modality=meta["modality"],
        )

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
