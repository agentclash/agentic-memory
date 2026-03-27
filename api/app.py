from __future__ import annotations

import json
import mimetypes
import os
from collections import deque
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import config
from events.bus import MemoryEvent
from events import EventBus
from models.base import MemoryRecord, normalize_modality
from models.episodic import EpisodicMemory
from models.semantic import SemanticMemory
from retrieval.retriever import UnifiedRetriever
from stores.episodic_store import EpisodicStore, EpisodicStoreError, MediaTooLargeError
from stores.media_store import MediaStore
from stores.semantic_store import SemanticStore
from utils.embeddings import TextEmbedder

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://memory.agentclash.dev",
]

DEFAULT_MEDIA_DIR = Path(os.getenv("MEMORY_MEDIA_DIR", config.MEDIA_STORAGE_PATH))
_SUPPORTED_FILE_MODALITIES = {"audio", "image", "video", "multimodal"}
_ALLOWED_MEDIA_TYPES = {"image", "audio", "video", "pdf"}


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


def _infer_media_contract(*, mime_type: str | None, filename: str | None) -> tuple[str, str] | None:
    guessed_mime = mime_type or mimetypes.guess_type(filename or "")[0]
    if guessed_mime:
        if guessed_mime.startswith("image/"):
            return "image", "image"
        if guessed_mime.startswith("audio/"):
            return "audio", "audio"
        if guessed_mime.startswith("video/"):
            return "video", "video"
        if guessed_mime == "application/pdf":
            return "multimodal", "pdf"

    suffix = Path(filename or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        return "image", "image"
    if suffix in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
        return "audio", "audio"
    if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        return "video", "video"
    if suffix == ".pdf":
        return "multimodal", "pdf"
    return None


def _validate_media_type(media_type: Any) -> str | None:
    if media_type is None:
        return None
    if not isinstance(media_type, str):
        raise ValueError("media_type must be a string if provided")

    resolved = media_type.strip().lower()
    if resolved not in _ALLOWED_MEDIA_TYPES:
        supported = ", ".join(sorted(_ALLOWED_MEDIA_TYPES))
        raise ValueError(f"Unsupported media_type '{media_type}'. Supported values: {supported}")
    return resolved


def _validate_emotional_profile(raw_profile: Any) -> dict[str, float]:
    if raw_profile is None:
        return {}
    if not isinstance(raw_profile, Mapping):
        raise ValueError("emotional_profile must be an object mapping strings to numbers")

    profile: dict[str, float] = {}
    for key, value in raw_profile.items():
        if not isinstance(key, str):
            raise ValueError("emotional_profile keys must be strings")
        if not isinstance(value, (int, float)):
            raise ValueError("emotional_profile values must be numeric")
        profile[key] = float(value)
    return profile


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
        "media_type": record.media_type,
        "text_description": record.text_description,
        "has_media": record.has_media,
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
                "emotional_profile": record.emotional_profile,
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
        media_root: Path | None = None,
        embedder: TextEmbedder | None = None,
    ):
        self.media_store = MediaStore(media_root or DEFAULT_MEDIA_DIR)
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

    def save_upload(self, upload: UploadFile, memory_id: str) -> tuple[str, str]:
        guessed_mime = upload.content_type or mimetypes.guess_type(upload.filename or "")[0] or "application/octet-stream"
        contents = upload.file.read()
        filename = upload.filename or "upload.bin"
        media_ref = self.media_store.store_bytes(contents, filename, memory_id)
        return media_ref, guessed_mime

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
    media_root: str | None = None,
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
        "media_root": Path(media_root) if media_root else None,
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
        try:
            media_type = _validate_media_type(payload.get("media_type"))
            record = SemanticMemory(
                content=payload["content"],
                importance=float(payload.get("importance", 0.5)),
                category=payload.get("category", "general"),
                confidence=float(payload.get("confidence", 1.0)),
                modality=payload.get("modality", "text"),
                media_ref=payload.get("media_ref"),
                media_type=media_type,
                text_description=payload.get("text_description"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        service().semantic_store.store(record)
        return {"record": _serialise_record(record)}

    @app.post("/api/memories/episodic/text")
    async def create_text_episode(payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("session_id"):
            raise HTTPException(status_code=400, detail="session_id is required")
        if not payload.get("text"):
            raise HTTPException(status_code=400, detail="text is required")
        try:
            emotional_profile = _validate_emotional_profile(payload.get("emotional_profile"))
            record = EpisodicMemory(
                content=payload["text"],
                session_id=payload["session_id"],
                turn_number=payload.get("turn_number"),
                participants=payload.get("participants", ["user", "agent"]),
                summary=payload.get("summary"),
                emotional_profile=emotional_profile,
                importance=float(payload.get("importance", 0.5)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        inferred_contract = _infer_media_contract(mime_type=file.content_type, filename=file.filename)
        inferred_modality = inferred_contract[0] if inferred_contract else None
        inferred_media_type = inferred_contract[1] if inferred_contract else None
        try:
            requested_modality = normalize_modality(modality) if modality is not None else None
            resolved_modality = requested_modality or normalize_modality(inferred_modality)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if requested_modality == "multimodal" and inferred_media_type != "pdf":
            raise HTTPException(
                status_code=400,
                detail="multimodal file uploads currently require a PDF file",
            )
        if requested_modality is not None and inferred_modality is not None and requested_modality != inferred_modality:
            raise HTTPException(
                status_code=400,
                detail="uploaded file does not match the requested modality",
            )
        if resolved_modality == "multimodal" and inferred_media_type != "pdf":
            raise HTTPException(
                status_code=400,
                detail="multimodal file uploads currently require a PDF file",
            )
        if resolved_modality not in _SUPPORTED_FILE_MODALITIES:
            raise HTTPException(
                status_code=400,
                detail="could not infer a supported modality from the uploaded file",
            )
        record = EpisodicMemory(
            content=content or f"{resolved_modality} episode from {file.filename}",
            session_id=session_id,
            modality=resolved_modality,
            media_type=inferred_media_type or resolved_modality,
            turn_number=turn_number,
            summary=summary,
            importance=importance,
        )
        media_ref, mime_type = service().save_upload(file, record.id)
        record.media_ref = media_ref
        record.source_mime_type = mime_type
        try:
            service().episodic_store.store(record)
        except MediaTooLargeError as exc:
            service().media_store.delete(media_ref)
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except EpisodicStoreError as exc:
            service().media_store.delete(media_ref)
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
