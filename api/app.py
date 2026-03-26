from __future__ import annotations

import json
import mimetypes
import os
from collections import deque
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from events.bus import MemoryEvent
from events import EventBus
from models.base import MemoryRecord
from models.episodic import EpisodicMemory
from models.semantic import SemanticMemory
from retrieval.retriever import UnifiedRetriever
from stores.episodic_store import EpisodicStore, EpisodicStoreError, MediaTooLargeError
from stores.semantic_store import SemanticStore
from utils.embeddings import TextEmbedder

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://memory.agentclash.dev",
]

DEFAULT_UPLOAD_DIR = Path(os.getenv("MEMORY_UPLOAD_DIR", "/tmp/agentic-memory-uploads"))
_SUPPORTED_MEDIA_MODALITIES = {"audio", "image", "video", "pdf"}


def _normalise_origins(origins: list[str] | None = None) -> list[str]:
    if origins is not None:
        return origins
    env_value = os.getenv("MEMORY_ALLOWED_ORIGINS")
    if not env_value:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in env_value.split(",") if origin.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _infer_media_modality(*, mime_type: str | None, filename: str | None) -> str | None:
    guessed_mime = mime_type or mimetypes.guess_type(filename or "")[0]
    if guessed_mime:
        if guessed_mime.startswith("image/"):
            return "image"
        if guessed_mime.startswith("audio/"):
            return "audio"
        if guessed_mime.startswith("video/"):
            return "video"
        if guessed_mime == "application/pdf":
            return "pdf"

    suffix = Path(filename or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        return "image"
    if suffix in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
        return "audio"
    if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        return "video"
    if suffix == ".pdf":
        return "pdf"
    return None


def _serialise_record(record: MemoryRecord) -> dict[str, Any]:
    payload = {
        "id": record.id,
        "memory_type": record.memory_type,
        "modality": record.modality,
        "content": record.content,
        "created_at": record.created_at.isoformat(),
        "last_accessed_at": record.last_accessed_at.isoformat() if record.last_accessed_at else None,
        "access_count": record.access_count,
        "importance": record.importance,
        "media_ref": record.media_ref,
    }
    if isinstance(record, SemanticMemory):
        payload.update(
            {
                "category": record.category,
                "confidence": record.confidence,
            }
        )
    if isinstance(record, EpisodicMemory):
        payload.update(
            {
                "session_id": record.session_id,
                "turn_number": record.turn_number,
                "participants": record.participants,
                "summary": record.summary,
                "emotional_valence": record.emotional_valence,
                "source_mime_type": record.source_mime_type,
            }
        )
    return payload


def _serialise_ranked_result(result) -> dict[str, Any]:
    return {
        "record": _serialise_record(result.record),
        "raw_similarity": result.raw_similarity,
        "recency_score": result.recency_score,
        "importance_score": result.importance_score,
        "final_score": result.final_score,
    }


class EventRecorder:
    def __init__(self, bus: EventBus, *, max_events: int = 200):
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        for event_type in ("memory.stored", "memory.retrieved", "memory.ranked", "memory.accessed"):
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


class MemoryAPIService:
    def __init__(
        self,
        *,
        chroma_path: str | None = None,
        upload_dir: Path | None = None,
        embedder: TextEmbedder | None = None,
    ):
        upload_root = upload_dir or DEFAULT_UPLOAD_DIR
        upload_root.mkdir(parents=True, exist_ok=True)
        self.upload_dir = upload_root
        self.bus = EventBus()
        original_chroma_path = config.CHROMA_DB_PATH
        try:
            if chroma_path is not None:
                config.CHROMA_DB_PATH = chroma_path
            self.semantic_store = SemanticStore(event_bus=self.bus, embedder=embedder)
            self.episodic_store = EpisodicStore(event_bus=self.bus, embedder=embedder)
        finally:
            config.CHROMA_DB_PATH = original_chroma_path
        self.retriever = UnifiedRetriever(
            stores={"semantic": self.semantic_store, "episodic": self.episodic_store},
            event_bus=self.bus,
        )
        self.events = EventRecorder(self.bus)

    def save_upload(self, upload: UploadFile) -> tuple[str, str]:
        suffix = Path(upload.filename or "upload").suffix
        target = self.upload_dir / f"{uuid4()}{suffix}"
        guessed_mime = upload.content_type or mimetypes.guess_type(upload.filename or "")[0] or "application/octet-stream"
        contents = upload.file.read()
        target.write_bytes(contents)
        return str(target), guessed_mime

    def overview(self) -> dict[str, Any]:
        semantic_count = self.semantic_store._collection.count()
        episodic_count = self.episodic_store._collection.count()
        recent = self.episodic_store.get_recent(5)
        return {
            "semantic_count": semantic_count,
            "episodic_count": episodic_count,
            "recent_sessions": sorted({record.session_id for record in recent}),
            "latest_events": self.events.snapshot(10),
        }


def create_app(
    *,
    chroma_path: str | None = None,
    upload_dir: str | None = None,
    allowed_origins: list[str] | None = None,
    embedder: TextEmbedder | None = None,
) -> FastAPI:
    app = FastAPI(title="Agentic Memory API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_normalise_origins(allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.service = None
    app.state.service_config = {
        "chroma_path": chroma_path,
        "upload_dir": Path(upload_dir) if upload_dir else None,
        "embedder": embedder,
    }

    def service() -> MemoryAPIService:
        if app.state.service is None:
            app.state.service = MemoryAPIService(**app.state.service_config)
        return app.state.service

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/overview")
    async def overview() -> dict[str, Any]:
        return service().overview()

    @app.get("/api/events")
    async def events(limit: int = Query(default=40, ge=1, le=200)) -> dict[str, Any]:
        return {"events": service().events.snapshot(limit)}

    @app.post("/api/memories/semantic")
    async def create_semantic_memory(payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("content"):
            raise HTTPException(status_code=400, detail="content is required")
        record = SemanticMemory(
            content=payload["content"],
            importance=float(payload.get("importance", 0.5)),
            category=payload.get("category", "general"),
            confidence=float(payload.get("confidence", 1.0)),
        )
        service().semantic_store.store(record)
        return {"record": _serialise_record(record)}

    @app.post("/api/memories/episodic/text")
    async def create_text_episode(payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("session_id"):
            raise HTTPException(status_code=400, detail="session_id is required")
        if not payload.get("text"):
            raise HTTPException(status_code=400, detail="text is required")
        record = EpisodicMemory(
            content=payload["text"],
            session_id=payload["session_id"],
            turn_number=payload.get("turn_number"),
            participants=payload.get("participants", ["user", "agent"]),
            summary=payload.get("summary"),
            importance=float(payload.get("importance", 0.5)),
        )
        service().episodic_store.store(record)
        return {"record": _serialise_record(record)}

    @app.post("/api/memories/episodic/file")
    async def create_file_episode(
        session_id: str = Form(...),
        modality: str | None = Form(default=None),
        content: str | None = Form(default=None),
        turn_number: int | None = Form(default=None),
        summary: str | None = Form(default=None),
        importance: float = Form(default=0.5),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        media_ref, mime_type = service().save_upload(file)
        inferred_modality = _infer_media_modality(mime_type=mime_type, filename=file.filename)
        resolved_modality = inferred_modality or modality
        if resolved_modality not in _SUPPORTED_MEDIA_MODALITIES:
            raise HTTPException(
                status_code=400,
                detail="could not infer a supported modality from the uploaded file",
            )
        record = EpisodicMemory(
            content=content or f"{resolved_modality} episode from {file.filename}",
            session_id=session_id,
            modality=resolved_modality,
            media_ref=media_ref,
            source_mime_type=mime_type,
            turn_number=turn_number,
            summary=summary,
            importance=importance,
        )
        try:
            service().episodic_store.store(record)
        except MediaTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except EpisodicStoreError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"record": _serialise_record(record)}

    @app.post("/api/retrieval/query")
    async def query(payload: dict[str, Any]) -> dict[str, Any]:
        text = payload.get("query", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="query is required")
        results = service().retriever.query(
            text,
            top_k=int(payload.get("top_k", 5)),
            memory_types=payload.get("memory_types"),
        )
        return {"results": [_serialise_ranked_result(result) for result in results]}

    @app.get("/api/episodes/recent")
    async def recent(n: int = Query(default=5, ge=1, le=50)) -> dict[str, Any]:
        records = service().retriever.query_recent(n)
        return {"records": [_serialise_record(record) for record in records]}

    @app.get("/api/episodes/session/{session_id}")
    async def by_session(session_id: str) -> dict[str, Any]:
        records = service().episodic_store.get_by_session(session_id)
        return {"records": [_serialise_record(record) for record in records]}

    @app.get("/api/episodes/time-range")
    async def by_time_range(start: datetime, end: datetime) -> dict[str, Any]:
        records = service().retriever.query_time_range(start, end)
        return {"records": [_serialise_record(record) for record in records]}

    return app


app = create_app()
