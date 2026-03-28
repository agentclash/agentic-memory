from __future__ import annotations

import json
import mimetypes
import os
import tempfile
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
from models.procedural import ProceduralMemory
from models.semantic import SemanticMemory
from retrieval.retriever import UnifiedRetriever
from stores.episodic_store import EpisodicStore, EpisodicStoreError, MediaTooLargeError
from stores.media_store import MediaStore
from stores.procedural_store import ProceduralMatch, ProceduralStore
from stores.semantic_store import SemanticStore
from utils.embeddings import EmbeddingProviderError, GeminiEmbedder, TextEmbedder

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
        if guessed_mime in {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}:
            return "image", "image"
        if guessed_mime in {
            "audio/mpeg",
            "audio/mp3",
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
            "audio/aiff",
            "audio/x-aiff",
            "audio/aac",
            "audio/flac",
            "audio/ogg",
        }:
            return "audio", "audio"
        if guessed_mime in {
            "video/mp4",
            "video/mpeg",
            "video/quicktime",
            "video/x-msvideo",
            "video/x-flv",
            "video/webm",
            "video/x-ms-wmv",
            "video/3gpp",
        }:
            return "video", "video"
        if guessed_mime == "application/pdf":
            return "multimodal", "pdf"

    suffix = Path(filename or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}:
        return "image", "image"
    if suffix in {".mp3", ".wav", ".aif", ".aiff", ".aac", ".flac", ".ogg"}:
        return "audio", "audio"
    if suffix in {".mp4", ".mpeg", ".mpg", ".mov", ".avi", ".flv", ".webm", ".wmv", ".3gp"}:
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


def _validate_related_ids(raw_related_ids: Any) -> list[str]:
    if raw_related_ids is None:
        return []
    if not isinstance(raw_related_ids, list):
        raise ValueError("related_ids must be an array of strings")

    related_ids: list[str] = []
    for value in raw_related_ids:
        if not isinstance(value, str):
            raise ValueError("related_ids must contain only strings")
        related_ids.append(value)
    return related_ids


def _validate_string_list(raw_value: Any, *, field_name: str, required: bool) -> list[str]:
    if raw_value is None:
        values: list[Any] = []
    elif not isinstance(raw_value, list):
        raise ValueError(f"{field_name} must be an array of strings")
    else:
        values = raw_value

    if required and not values:
        raise ValueError(f"{field_name} must contain at least one entry")

    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain only non-blank strings")
        cleaned.append(value)
    return cleaned


def _parse_memory_types(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values:
        return None

    supported = {"semantic", "episodic", "procedural"}
    invalid = [value for value in values if value not in supported]
    if invalid:
        raise ValueError(
            f"Unsupported memory_types {invalid}. Supported values: {', '.join(sorted(supported))}"
        )
    return values


def _validate_query_upload(upload: UploadFile, *, modality: str) -> str:
    inferred_contract = _infer_media_contract(
        mime_type=upload.content_type,
        filename=upload.filename,
    )
    if inferred_contract is None:
        raise ValueError(f"query-by-{modality} requires a supported {modality} file upload")

    inferred_modality, _ = inferred_contract
    if inferred_modality != modality:
        raise ValueError(f"query-by-{modality} requires a supported {modality} file upload")

    return upload.content_type or mimetypes.guess_type(upload.filename or "")[0] or "application/octet-stream"


def _cleanup_owned_media(service: MemoryAPIService, media_ref: str | None) -> None:
    if media_ref and service.media_store.owns(media_ref):
        service.media_store.delete(media_ref)


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
                "domain": record.domain,
                "confidence": record.confidence,
                "supersedes": record.supersedes,
                "related_ids": record.related_ids,
                "has_visual": record.has_visual,
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
    if isinstance(record, ProceduralMemory):
        payload.update(
            {
                "steps": record.steps,
                "preconditions": record.preconditions,
                "success_count": record.success_count,
                "failure_count": record.failure_count,
                "total_outcomes": record.total_outcomes,
                "success_rate": record.success_rate,
                "wilson_score": record.wilson_score,
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


def _serialise_procedural_match(match: ProceduralMatch) -> dict[str, Any]:
    return {
        "record": _serialise_record(match.record),
        "similarity": match.similarity,
        "wilson_score": match.wilson_score,
        "combined_score": match.combined_score,
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
        self.embedder = embedder or GeminiEmbedder()
        original_chroma_path = config.CHROMA_DB_PATH
        try:
            if chroma_path is not None:
                config.CHROMA_DB_PATH = chroma_path
            self.semantic_store = SemanticStore(
                event_bus=self.bus,
                embedder=self.embedder,
                media_store=self.media_store,
            )
            self.episodic_store = EpisodicStore(
                event_bus=self.bus,
                embedder=self.embedder,
                media_store=self.media_store,
            )
            self.procedural_store = ProceduralStore(
                event_bus=self.bus,
                embedder=self.embedder,
                media_store=self.media_store,
            )
        finally:
            config.CHROMA_DB_PATH = original_chroma_path
        self.retriever = UnifiedRetriever(
            stores={
                "semantic": self.semantic_store,
                "episodic": self.episodic_store,
                "procedural": self.procedural_store,
            },
            event_bus=self.bus,
        )
        self.events = EventRecorder(self.bus)

    def save_upload(self, upload: UploadFile, memory_id: str) -> tuple[str, str]:
        guessed_mime = upload.content_type or mimetypes.guess_type(upload.filename or "")[0] or "application/octet-stream"
        contents = upload.file.read()
        filename = upload.filename or "upload.bin"
        media_ref = self.media_store.store_bytes(contents, filename, memory_id)
        return media_ref, guessed_mime

    def save_query_upload(self, upload: UploadFile) -> tuple[str, str]:
        guessed_mime = upload.content_type or mimetypes.guess_type(upload.filename or "")[0] or "application/octet-stream"
        suffix = Path(upload.filename or "upload.bin").suffix
        contents = upload.file.read()
        handle = tempfile.NamedTemporaryFile(prefix="memory_query_", suffix=suffix, delete=False)
        try:
            handle.write(contents)
            handle.flush()
        finally:
            handle.close()
        return handle.name, guessed_mime

    def overview(self) -> dict[str, Any]:
        semantic_count = self.semantic_store._collection.count()
        episodic_count = self.episodic_store._collection.count()
        procedural_count = self.procedural_store._collection.count()
        recent = self.episodic_store.get_recent(5)
        return {
            "semantic_count": semantic_count,
            "episodic_count": episodic_count,
            "procedural_count": procedural_count,
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
            modality = payload.get("modality", "text")
            resolved_modality = normalize_modality(modality)
            media_ref = payload.get("media_ref")
            inferred_contract = _infer_media_contract(mime_type=None, filename=media_ref)
            inferred_media_type = inferred_contract[1] if inferred_contract else None
            media_type = _validate_media_type(payload.get("media_type")) or inferred_media_type
            if resolved_modality != "text" and not media_ref:
                raise ValueError("media_ref is required when modality is not text")
            if resolved_modality == "multimodal" and media_type is None:
                raise ValueError("multimodal semantic memory requires a supported media_type")
            if (
                resolved_modality in {"image", "audio", "video"}
                and media_type is not None
                and media_type != resolved_modality
            ):
                raise ValueError(
                    f"media_type '{media_type}' does not match modality '{resolved_modality}'"
                )
            related_ids = _validate_related_ids(payload.get("related_ids"))
            record = SemanticMemory(
                content=payload["content"],
                importance=float(payload.get("importance", 0.5)),
                category=payload.get("category", "general"),
                domain=payload.get("domain"),
                confidence=float(payload.get("confidence", 1.0)),
                supersedes=payload.get("supersedes"),
                related_ids=related_ids,
                has_visual=bool(payload.get("has_visual", False)),
                modality=resolved_modality,
                media_ref=media_ref,
                media_type=media_type,
                text_description=payload.get("text_description"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        active_service = service()
        try:
            active_service.semantic_store.store(record)
        except (FileNotFoundError, ValueError) as exc:
            _cleanup_owned_media(active_service, record.media_ref)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EmbeddingProviderError as exc:
            _cleanup_owned_media(active_service, record.media_ref)
            raise HTTPException(
                status_code=502,
                detail="Gemini embedding provider failed after retries",
            ) from exc
        except Exception:
            _cleanup_owned_media(active_service, record.media_ref)
            raise
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
        if requested_modality == "multimodal" and inferred_media_type is None:
            raise HTTPException(
                status_code=400,
                detail="multimodal file uploads require a supported image, audio, video, or PDF file",
            )
        if (
            requested_modality is not None
            and inferred_modality is not None
            and requested_modality != inferred_modality
            and requested_modality != "multimodal"
        ):
            raise HTTPException(
                status_code=400,
                detail="uploaded file does not match the requested modality",
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

    @app.post("/api/memories/procedural")
    async def create_procedural_memory(payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("content"):
            raise HTTPException(status_code=400, detail="content is required")
        try:
            steps = _validate_string_list(payload.get("steps"), field_name="steps", required=True)
            preconditions = _validate_string_list(
                payload.get("preconditions"),
                field_name="preconditions",
                required=False,
            )
            record = ProceduralMemory(
                content=payload["content"],
                steps=steps,
                preconditions=preconditions,
                importance=float(payload.get("importance", 0.5)),
                source=payload.get("source"),
                metadata=payload.get("metadata") or {},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        service().procedural_store.store(record)
        return {"record": _serialise_record(record)}

    @app.post("/api/memories/procedural/file")
    async def create_file_procedure(
        content: str = Form(...),
        steps: list[str] = Form(...),
        preconditions: list[str] | None = Form(default=None),
        modality: str | None = Form(default=None),
        media_type: str | None = Form(default=None),
        text_description: str | None = Form(default=None),
        importance: float = Form(default=0.5),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        inferred_contract = _infer_media_contract(mime_type=file.content_type, filename=file.filename)
        inferred_modality = inferred_contract[0] if inferred_contract else None
        inferred_media_type = inferred_contract[1] if inferred_contract else None
        requested_media_type: str | None = None
        try:
            parsed_steps = _validate_string_list(steps, field_name="steps", required=True)
            parsed_preconditions = _validate_string_list(
                preconditions,
                field_name="preconditions",
                required=False,
            )
            requested_modality = normalize_modality(modality) if modality is not None else None
            resolved_modality = requested_modality or normalize_modality(inferred_modality)
            requested_media_type = _validate_media_type(media_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if requested_modality == "multimodal" and inferred_media_type is None and requested_media_type is None:
            raise HTTPException(
                status_code=400,
                detail="multimodal file uploads require a supported image, audio, video, or PDF file",
            )
        if (
            requested_modality is not None
            and inferred_modality is not None
            and requested_modality != inferred_modality
            and requested_modality != "multimodal"
        ):
            raise HTTPException(
                status_code=400,
                detail="uploaded file does not match the requested modality",
            )
        if resolved_modality not in _SUPPORTED_FILE_MODALITIES:
            raise HTTPException(
                status_code=400,
                detail="could not infer a supported modality from the uploaded file",
            )
        resolved_media_type = requested_media_type or inferred_media_type or resolved_modality
        record = ProceduralMemory(
            content=content,
            steps=parsed_steps,
            preconditions=parsed_preconditions,
            modality=resolved_modality,
            media_type=resolved_media_type,
            text_description=text_description,
            importance=importance,
        )
        media_ref, _ = service().save_upload(file, record.id)
        record.media_ref = media_ref
        try:
            service().procedural_store.store(record)
        except (FileNotFoundError, ValueError) as exc:
            service().media_store.delete(media_ref)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EmbeddingProviderError as exc:
            service().media_store.delete(media_ref)
            raise HTTPException(
                status_code=502,
                detail="Gemini embedding provider failed after retries",
            ) from exc
        return {"record": _serialise_record(record)}

    @app.post("/api/memories/procedural/{record_id}/outcome")
    async def record_procedural_outcome(record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if "success" not in payload or not isinstance(payload["success"], bool):
            raise HTTPException(status_code=400, detail="success must be provided as a boolean")
        active_service = service()
        record = active_service.procedural_store.get_by_id(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="procedural memory not found")
        active_service.procedural_store.record_outcome(record_id, payload["success"])
        updated = active_service.procedural_store.get_by_id(record_id)
        return {"record": _serialise_record(updated)}

    @app.post("/api/retrieval/best-procedures")
    async def best_procedures(payload: dict[str, Any]) -> dict[str, Any]:
        task = payload.get("task", "").strip()
        if not task:
            raise HTTPException(status_code=400, detail="task is required")
        matches = service().procedural_store.get_best_procedure_matches(
            task,
            top_k=int(payload.get("top_k", 3)),
        )
        return {"results": [_serialise_procedural_match(match) for match in matches]}

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

    @app.post("/api/retrieval/query-by-image")
    async def query_by_image(
        file: UploadFile = File(...),
        top_k: int = Form(default=5),
        memory_types: str | None = Form(default=None),
    ) -> dict[str, Any]:
        active_service = service()
        query_path = None
        try:
            parsed_memory_types = _parse_memory_types(memory_types)
            _validate_query_upload(file, modality="image")
            query_path, mime_type = active_service.save_query_upload(file)
            vector = active_service.embedder.embed_image(query_path, mime_type=mime_type)
            results = active_service.retriever.query_by_vector(
                vector,
                top_k=top_k,
                memory_types=parsed_memory_types,
                metadata={"source_modality": "image"},
            )
            return {
                "query_type": "vector",
                "source_modality": "image",
                "results": [_serialise_ranked_result(result) for result in results],
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EmbeddingProviderError as exc:
            raise HTTPException(
                status_code=502,
                detail="Gemini embedding provider failed after retries",
            ) from exc
        finally:
            if query_path and os.path.exists(query_path):
                os.remove(query_path)

    @app.post("/api/retrieval/query-by-audio")
    async def query_by_audio(
        file: UploadFile = File(...),
        top_k: int = Form(default=5),
        memory_types: str | None = Form(default=None),
    ) -> dict[str, Any]:
        active_service = service()
        query_path = None
        try:
            parsed_memory_types = _parse_memory_types(memory_types)
            _validate_query_upload(file, modality="audio")
            query_path, mime_type = active_service.save_query_upload(file)
            vector = active_service.embedder.embed_audio(query_path, mime_type=mime_type)
            results = active_service.retriever.query_by_vector(
                vector,
                top_k=top_k,
                memory_types=parsed_memory_types,
                metadata={"source_modality": "audio"},
            )
            return {
                "query_type": "vector",
                "source_modality": "audio",
                "results": [_serialise_ranked_result(result) for result in results],
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EmbeddingProviderError as exc:
            raise HTTPException(
                status_code=502,
                detail="Gemini embedding provider failed after retries",
            ) from exc
        finally:
            if query_path and os.path.exists(query_path):
                os.remove(query_path)

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
