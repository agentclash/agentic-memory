import json
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chromadb

import config
from events.bus import EventBus
from models.procedural import ProceduralMemory
from stores.base import BaseStore
from stores.media_store import MediaStore
from utils.embeddings import GeminiEmbedder, TextEmbedder

_PROCEDURAL_SIMILARITY_WEIGHT = 0.5
_PROCEDURAL_WILSON_WEIGHT = 0.5
_CANDIDATE_MULTIPLIER = 3


@dataclass(frozen=True, slots=True)
class ProceduralMatch:
    record: ProceduralMemory
    similarity: float
    wilson_score: float
    combined_score: float


class ProceduralStore(BaseStore):
    """ChromaDB-backed store for procedural memories."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        embedder: TextEmbedder | None = None,
        media_store: MediaStore | None = None,
    ):
        super().__init__(event_bus=event_bus)
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self._collection = client.get_or_create_collection(
            name="procedural_memories",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder or GeminiEmbedder()
        self._media_store = media_store

    def store(self, record: ProceduralMemory) -> str:
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
                    "step_count": len(record.steps),
                    "precondition_count": len(record.preconditions),
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

    def get_by_id(self, record_id: str) -> ProceduralMemory | None:
        result = self._collection.get(
            ids=[record_id],
            include=["embeddings", "documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return self._from_result(result, 0)

    def get_all_records(self, include_embeddings: bool = False) -> list[ProceduralMemory]:
        return self._all_records(include_embeddings=include_embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[ProceduralMemory, float]]:
        query_embedding = self._embedder.embed_query(query)
        return self.retrieve_by_vector(query_embedding, top_k=top_k)

    def retrieve_by_vector(
        self,
        vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[ProceduralMemory, float]]:
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["embeddings", "documents", "metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        pairs = []
        for index in range(len(result["ids"][0])):
            record = self._from_query_result(result, index)
            similarity = 1.0 - result["distances"][0][index]
            pairs.append((record, similarity))
        return pairs

    def update_access(self, record_id: str) -> None:
        now = datetime.now(timezone.utc)
        result = self._collection.get(ids=[record_id], include=["metadatas"])
        if not result["ids"]:
            return
        meta = result["metadatas"][0]
        meta["access_count"] = int(meta.get("access_count", 0)) + 1
        meta["last_accessed_at"] = now.isoformat()
        self._collection.update(ids=[record_id], metadatas=[meta])

    def delete(self, record_id: str) -> None:
        self._collection.delete(ids=[record_id])

    def replace(self, record: ProceduralMemory) -> None:
        if record.embedding is None:
            raise ValueError("replace(record) requires record.embedding to be set")
        self._collection.update(
            ids=[record.id],
            embeddings=[record.embedding],
            documents=[record.content],
            metadatas=[self._to_metadata(record)],
        )

    def record_outcome(self, record_id: str, success: bool) -> None:
        record = self.get_by_id(record_id)
        if record is None:
            return
        record.record_outcome(success)
        self.replace(record)

    def get_best_procedures(
        self,
        task: str,
        top_k: int = 3,
    ) -> list[tuple[ProceduralMemory, float]]:
        return [
            (match.record, match.combined_score)
            for match in self.get_best_procedure_matches(task, top_k=top_k)
        ]

    def get_best_procedure_matches(
        self,
        task: str,
        top_k: int = 3,
    ) -> list[ProceduralMatch]:
        if top_k <= 0:
            return []
        candidates = self.retrieve(task, top_k=top_k * _CANDIDATE_MULTIPLIER)
        ranked = [
            ProceduralMatch(
                record=record,
                similarity=similarity,
                wilson_score=record.wilson_score,
                # TODO: v2 should consider an explicit exploration policy so
                # highly relevant untested procedures are surfaced intentionally
                # instead of only through this simpler conditional weighting.
                combined_score=(
                    similarity
                    if record.total_outcomes == 0
                    else (
                        similarity * _PROCEDURAL_SIMILARITY_WEIGHT
                        + record.wilson_score * _PROCEDURAL_WILSON_WEIGHT
                    )
                ),
            )
            for record, similarity in candidates
        ]
        return sorted(ranked, key=lambda match: match.combined_score, reverse=True)[:top_k]

    def _to_metadata(self, record: ProceduralMemory) -> dict:
        return {
            "memory_type": record.memory_type,
            "modality": record.modality,
            "created_at": record.created_at.isoformat(),
            "last_accessed_at": record.last_accessed_at.isoformat() if record.last_accessed_at else "",
            "access_count": record.access_count,
            "importance": record.importance,
            "source": record.source or "",
            "metadata_json": json.dumps(record.metadata, sort_keys=True),
            "media_ref": record.media_ref or "",
            "media_type": record.media_type or "",
            "text_description": record.text_description or "",
            "steps_json": json.dumps(record.steps),
            "preconditions_json": json.dumps(record.preconditions),
            "success_count": record.success_count,
            "failure_count": record.failure_count,
        }

    def _all_records(self, *, include_embeddings: bool) -> list[ProceduralMemory]:
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.insert(0, "embeddings")

        result = self._collection.get(include=include)
        if not result["ids"]:
            return []
        return [self._from_result(result, index) for index in range(len(result["ids"]))]

    def _build_record(self, doc: str, record_id: str, embedding, meta: dict) -> ProceduralMemory:
        last_accessed = meta.get("last_accessed_at", "")
        return ProceduralMemory(
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
            media_ref=meta.get("media_ref") or None,
            media_type=meta.get("media_type") or None,
            text_description=meta.get("text_description") or None,
            steps=json.loads(meta.get("steps_json", "[]")),
            preconditions=json.loads(meta.get("preconditions_json", "[]")),
            success_count=int(meta.get("success_count", 0)),
            failure_count=int(meta.get("failure_count", 0)),
        )

    def _embed_record(self, record: ProceduralMemory) -> list[float]:
        if record.modality == "text":
            return self._embedder.embed_text(record.content)

        media_path = self._require_media_path(record)
        media_type = self._resolve_multimodal_media_type(record, media_path)
        return self._embed_multimodal(
            media_type,
            media_path,
            self._text_context(record),
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

    def _text_context(self, record: ProceduralMemory) -> str:
        parts = [record.content]
        if record.text_description:
            parts.append(record.text_description)
        return "\n".join(part for part in parts if part)

    def _ensure_owned_media(self, record: ProceduralMemory) -> str | None:
        if not record.media_ref or self._media_store is None:
            return None
        record.media_ref, copied = self._media_store.ensure_owned(record.media_ref, record.id)
        return record.media_ref if copied else None

    def _require_media_path(self, record: ProceduralMemory) -> Path:
        if not record.media_ref:
            raise ValueError(f"Procedural {record.modality} memory requires media_ref")
        media_path = Path(record.media_ref)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {record.media_ref}")
        return media_path

    def _resolve_multimodal_media_type(self, record: ProceduralMemory, media_path: Path) -> str:
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

    def _from_result(self, result: dict, index: int) -> ProceduralMemory:
        embeddings = result.get("embeddings")
        return self._build_record(
            result["documents"][index],
            result["ids"][index],
            embeddings[index] if embeddings is not None else None,
            result["metadatas"][index],
        )

    def _from_query_result(self, result: dict, index: int) -> ProceduralMemory:
        return self._build_record(
            result["documents"][0][index],
            result["ids"][0][index],
            result["embeddings"][0][index],
            result["metadatas"][0][index],
        )
