from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import mimetypes
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from mcp_server.schemas import (
    DEFAULT_MAX_CONTRADICTIONS,
    ForgettingCycleRequest,
    ForgettingReportModel,
    GetMemoryRequest,
    GetMemoryResponse,
    MCPToolError,
    MediaInputModel,
    MemoryOverviewModel,
    RecallEpisodesRequestAdapter,
    RecallEpisodesResponse,
    RecallMemoriesRequest,
    RecallMemoriesResponse,
    RecallProceduresRequest,
    RecallProceduresResponse,
    RecordProcedureOutcomeRequest,
    RecordProcedureOutcomeResponse,
    RememberEpisodeRequest,
    RememberEpisodeResponse,
    RememberFactRequest,
    RememberFactResponse,
    RememberProcedureRequest,
    RememberProcedureResponse,
    ResolveContradictionRequest,
    ResolveContradictionResponse,
    ToolErrorModel,
    get_memory_by_type,
    serialise_forgetting_report,
    serialise_memory_overview,
    serialise_procedural_match,
    serialise_ranked_result,
    serialise_record,
    serialise_remember_fact_response,
)
from models.base import normalize_modality
from models.episodic import EpisodicMemory
from models.procedural import ProceduralMemory
from models.semantic import SemanticMemory
from runtime import MemoryRuntime, build_runtime
from stores.episodic_store import EpisodicStoreError, MediaTooLargeError
from stores.media_store import MediaStore
from utils.embeddings import EmbeddingProviderError, TextEmbedder

logger = logging.getLogger(__name__)

_TRANSPORTS = {"stdio", "streamable-http"}
_MCP_SERVER_NAME = "agentic-memory"


@dataclass(frozen=True, slots=True)
class ServerConfig:
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000
    chroma_path: str | None = None
    media_root: str | None = None
    embedding_dimensions: int | None = None
    streamable_http_path: str = "/mcp"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
        port = os.getenv("MEMORY_MCP_PORT")
        return cls(
            transport=os.getenv("MEMORY_MCP_TRANSPORT", "stdio").strip().lower(),
            host=os.getenv("MEMORY_MCP_HOST", "127.0.0.1").strip(),
            port=int(port) if port else 8000,
            chroma_path=os.getenv("MEMORY_CHROMA_PATH") or None,
            media_root=os.getenv("MEMORY_MEDIA_DIR") or None,
            embedding_dimensions=int(embedding_dimensions) if embedding_dimensions else None,
            streamable_http_path=os.getenv("MEMORY_MCP_PATH", "/mcp").strip() or "/mcp",
        )

    def validate(self) -> "ServerConfig":
        if self.transport not in _TRANSPORTS:
            raise ValueError(
                "MEMORY_MCP_TRANSPORT must be one of: stdio, streamable-http"
            )
        if self.port <= 0:
            raise ValueError("MEMORY_MCP_PORT must be a positive integer")
        if not self.streamable_http_path.startswith("/"):
            raise ValueError("MEMORY_MCP_PATH must start with '/'")
        if self.streamable_http_path != "/" and self.streamable_http_path.endswith("/"):
            raise ValueError("MEMORY_MCP_PATH must not end with '/'")

        if self.transport == "streamable-http":
            for label, raw_path in (
                ("MEMORY_CHROMA_PATH", self.chroma_path or config.CHROMA_DB_PATH),
                ("MEMORY_MEDIA_DIR", self.media_root or config.MEDIA_STORAGE_PATH),
            ):
                if not Path(raw_path).is_absolute():
                    logger.warning(
                        "%s is relative (%s). This is safe for local dev but risky for remote "
                        "streamable-http deployments due to working-directory drift.",
                        label,
                        raw_path,
                    )
        return self


@dataclass(frozen=True, slots=True)
class PreparedMedia:
    path: Path
    media_type: str
    cleanup: bool = False


def _map_exception(exc: Exception) -> MCPToolError:
    if isinstance(exc, MCPToolError):
        return exc
    if isinstance(exc, MediaTooLargeError):
        return MCPToolError(
            code="media_too_large",
            message=str(exc),
            details={"exception_type": exc.__class__.__name__},
        )
    if isinstance(exc, (FileNotFoundError, ValueError, EpisodicStoreError, binascii.Error)):
        return MCPToolError(
            code="invalid_request",
            message=str(exc),
            details={"exception_type": exc.__class__.__name__},
        )
    if isinstance(exc, EmbeddingProviderError):
        return MCPToolError(
            code="dependency_unavailable",
            message="Gemini embedding provider failed after retries",
            retryable=True,
            details={"exception_type": exc.__class__.__name__},
        )
    logger.exception("Unhandled MCP server error", exc_info=exc)
    return MCPToolError(
        code="internal_error",
        message=str(exc) or exc.__class__.__name__,
        details={"exception_type": exc.__class__.__name__},
    )


def _safe_contradiction_lookup(
    runtime: MemoryRuntime,
    record: SemanticMemory,
    *,
    max_contradictions: int = DEFAULT_MAX_CONTRADICTIONS,
) -> RememberFactResponse:
    try:
        candidates = runtime.contradiction_detector.find_potential_contradictions(
            record,
            top_k=max_contradictions,
        )
        return serialise_remember_fact_response(
            record=record,
            contradiction_status="completed",
            contradiction_candidates=candidates,
            max_contradictions=max_contradictions,
        )
    except ValueError as exc:
        return serialise_remember_fact_response(
            record=record,
            contradiction_status="skipped",
            contradiction_candidates=[],
            contradiction_detail=str(exc),
            max_contradictions=max_contradictions,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("Contradiction lookup failed for semantic record %s", record.id)
        return serialise_remember_fact_response(
            record=record,
            contradiction_status="error",
            contradiction_candidates=[],
            contradiction_detail=str(exc),
            max_contradictions=max_contradictions,
        )


class MemoryMCPServer:
    def __init__(
        self,
        *,
        config: ServerConfig,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self._config = config.validate()
        self._embedder = embedder
        self._runtime: MemoryRuntime | None = None
        self._forgetting_lock = asyncio.Lock()

    @property
    def config(self) -> ServerConfig:
        return self._config

    def runtime(self) -> MemoryRuntime:
        if self._runtime is None:
            self._runtime = build_runtime(
                chroma_path=self._config.chroma_path,
                media_root=self._config.media_root,
                embedder=self._embedder,
                embedding_dimensions=self._config.embedding_dimensions,
            )
        return self._runtime

    def validate_startup(self) -> None:
        self.runtime()

    def overview(self) -> MemoryOverviewModel:
        runtime = self.runtime()
        payload = {
            "semantic_count": runtime.semantic_store._collection.count(),
            "episodic_count": runtime.episodic_store._collection.count(),
            "procedural_count": runtime.procedural_store._collection.count(),
            "recent_sessions": sorted(
                {record.session_id for record in runtime.episodic_store.get_recent(5)}
            ),
            "latest_events": runtime.events.snapshot(10),
        }
        return serialise_memory_overview(payload)

    def remember_fact(self, request: RememberFactRequest) -> RememberFactResponse:
        runtime = self.runtime()
        prepared_media = self._prepare_media(request.media, request.media_type)
        try:
            modality, media_type = self._resolve_media_contract(
                modality=request.modality,
                prepared_media=prepared_media,
                requested_media_type=request.media_type,
            )
            record = SemanticMemory(
                content=request.content,
                importance=request.importance,
                category=request.category,
                domain=request.domain,
                confidence=request.confidence,
                supersedes=request.supersedes,
                related_ids=list(request.related_ids),
                has_visual=request.has_visual,
                modality=modality,
                media_ref=str(prepared_media.path) if prepared_media else None,
                media_type=media_type,
                text_description=request.text_description,
            )
            runtime.semantic_store.store(record)
            return _safe_contradiction_lookup(
                runtime,
                record,
                max_contradictions=request.max_contradictions,
            )
        finally:
            self._cleanup_prepared_media(prepared_media)

    def remember_episode(self, request: RememberEpisodeRequest) -> RememberEpisodeResponse:
        runtime = self.runtime()
        prepared_media = self._prepare_media(request.media, request.media_type)
        try:
            modality, media_type = self._resolve_media_contract(
                modality=request.modality,
                prepared_media=prepared_media,
                requested_media_type=request.media_type,
            )
            record = EpisodicMemory(
                content=request.text,
                session_id=request.session_id,
                turn_number=request.turn_number,
                participants=list(request.participants),
                summary=request.summary,
                emotional_profile=dict(request.emotional_profile),
                emotional_valence=request.emotional_valence,
                importance=request.importance,
                modality=modality,
                media_ref=str(prepared_media.path) if prepared_media else None,
                media_type=media_type,
                source_mime_type=mimetypes.guess_type(prepared_media.path.name)[0]
                if prepared_media
                else request.source_mime_type,
            )
            runtime.episodic_store.store(record)
            return RememberEpisodeResponse(record=serialise_record(record))
        finally:
            self._cleanup_prepared_media(prepared_media)

    def remember_procedure(self, request: RememberProcedureRequest) -> RememberProcedureResponse:
        runtime = self.runtime()
        prepared_media = self._prepare_media(request.media, request.media_type)
        try:
            modality, media_type = self._resolve_media_contract(
                modality=request.modality,
                prepared_media=prepared_media,
                requested_media_type=request.media_type,
            )
            record = ProceduralMemory(
                content=request.content,
                steps=list(request.steps),
                preconditions=list(request.preconditions),
                importance=request.importance,
                modality=modality,
                media_ref=str(prepared_media.path) if prepared_media else None,
                media_type=media_type,
                text_description=request.text_description,
            )
            runtime.procedural_store.store(record)
            return RememberProcedureResponse(record=serialise_record(record))
        finally:
            self._cleanup_prepared_media(prepared_media)

    def resolve_contradiction(
        self,
        request: ResolveContradictionRequest,
    ) -> ResolveContradictionResponse:
        self.runtime().contradiction_detector.resolve_supersession(
            superseded_id=request.supersede_id,
            kept_id=request.keep_id,
        )
        return ResolveContradictionResponse(
            kept_id=request.keep_id,
            superseded_id=request.supersede_id,
        )

    def record_procedure_outcome(
        self,
        request: RecordProcedureOutcomeRequest,
    ) -> RecordProcedureOutcomeResponse:
        runtime = self.runtime()
        record = runtime.procedural_store.get_by_id(request.record_id)
        if record is None:
            raise ValueError(f"Procedural memory '{request.record_id}' not found")
        runtime.procedural_store.record_outcome(request.record_id, request.success)
        updated = runtime.procedural_store.get_by_id(request.record_id)
        if updated is None:
            raise ValueError(f"Procedural memory '{request.record_id}' not found after update")
        return RecordProcedureOutcomeResponse(record=serialise_record(updated))

    def recall_memories(self, request: RecallMemoriesRequest) -> RecallMemoriesResponse:
        results = self.runtime().retriever.query(
            request.query,
            top_k=request.top_k,
            memory_types=list(request.memory_types) if request.memory_types else None,
        )
        return RecallMemoriesResponse(
            results=[serialise_ranked_result(result) for result in results]
        )

    def recall_procedures(self, request: RecallProceduresRequest) -> RecallProceduresResponse:
        matches = self.runtime().procedural_store.get_best_procedure_matches(
            request.task,
            top_k=request.top_k,
        )
        return RecallProceduresResponse(
            results=[serialise_procedural_match(match) for match in matches]
        )

    def recall_episodes(self, payload: dict[str, Any]) -> RecallEpisodesResponse:
        runtime = self.runtime()
        request = RecallEpisodesRequestAdapter.validate_python(payload)
        if request.mode == "recent":
            records = runtime.retriever.query_recent(request.limit)
        elif request.mode == "session":
            records = runtime.episodic_store.get_by_session(request.session_id)
        else:
            records = runtime.retriever.query_time_range(request.start, request.end)

        return RecallEpisodesResponse(
            mode=request.mode,
            records=[serialise_record(record) for record in records],
        )

    def get_memory(self, request: GetMemoryRequest) -> GetMemoryResponse:
        record = get_memory_by_type(self.runtime(), request)
        return GetMemoryResponse(
            record=serialise_record(record) if record is not None else None
        )

    async def preview_forgetting_cycle(
        self,
        request: ForgettingCycleRequest,
    ) -> ForgettingReportModel:
        return await self._run_forgetting_cycle(request=request, dry_run=True)

    async def run_forgetting_cycle(
        self,
        request: ForgettingCycleRequest,
    ) -> ForgettingReportModel:
        return await self._run_forgetting_cycle(request=request, dry_run=False)

    async def _run_forgetting_cycle(
        self,
        *,
        request: ForgettingCycleRequest,
        dry_run: bool,
    ) -> ForgettingReportModel:
        if self._forgetting_lock.locked():
            return ForgettingReportModel(status="already_running", dry_run=dry_run)

        async with self._forgetting_lock:
            report = self.runtime().forgetting_service.run_cycle(dry_run=dry_run)
            return serialise_forgetting_report(report, max_decisions=request.max_decisions)

    def _prepare_media(
        self,
        media: MediaInputModel | None,
        requested_media_type: str | None,
    ) -> PreparedMedia | None:
        if media is None:
            return None
        if media.file_path:
            path = Path(media.file_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Media file not found: {path}")
            if not path.is_file():
                raise ValueError(f"Media path is not a file: {path}")
            if not os.access(path, os.R_OK):
                raise ValueError(f"Media file is not readable: {path}")
            media_type = MediaStore.resolve_media_type(path, requested_media_type)
            return PreparedMedia(path=path, media_type=media_type)

        inline = media.inline_content
        if inline is None:
            raise ValueError("Media input requires file_path or inline_content")

        try:
            payload = base64.b64decode(inline.inline_base64, validate=True)
        except binascii.Error as exc:
            raise ValueError("inline_base64 must be valid base64") from exc
        if not payload:
            raise ValueError("Cannot store empty media payload")

        suffix = Path(inline.filename or "").suffix
        if not suffix:
            suffix = mimetypes.guess_extension(inline.mime_type) or ""
        handle = tempfile.NamedTemporaryFile(
            prefix="agentic_memory_mcp_",
            suffix=suffix,
            delete=False,
        )
        try:
            handle.write(payload)
            handle.flush()
        finally:
            handle.close()

        path = Path(handle.name)
        media_type = MediaStore.resolve_media_type(path, requested_media_type)
        return PreparedMedia(path=path, media_type=media_type, cleanup=True)

    @staticmethod
    def _cleanup_prepared_media(prepared_media: PreparedMedia | None) -> None:
        if prepared_media is None or not prepared_media.cleanup:
            return
        try:
            prepared_media.path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove temporary media file %s", prepared_media.path)

    @staticmethod
    def _resolve_media_contract(
        *,
        modality: str,
        prepared_media: PreparedMedia | None,
        requested_media_type: str | None,
    ) -> tuple[str, str | None]:
        resolved_modality = normalize_modality(modality)
        if prepared_media is None:
            if resolved_modality != "text":
                raise ValueError("media is required when modality is not text")
            return resolved_modality, requested_media_type

        resolved_media_type = prepared_media.media_type
        if requested_media_type and requested_media_type != resolved_media_type:
            raise ValueError(
                f"media_type '{requested_media_type}' does not match detected media type "
                f"'{resolved_media_type}'"
            )

        inferred_modality = "multimodal" if resolved_media_type == "pdf" else resolved_media_type
        if resolved_modality == "text":
            resolved_modality = inferred_modality
        elif resolved_modality == "multimodal" and resolved_media_type != "pdf":
            pass
        elif resolved_modality != inferred_modality:
            raise ValueError(
                f"modality '{resolved_modality}' does not match media type '{resolved_media_type}'"
            )

        return resolved_modality, resolved_media_type


def _load_fastmcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - import depends on optional dependency
        raise RuntimeError(
            "The 'mcp' package is required to run the MCP server. "
            "Install it with: pip install \"mcp[cli]\""
        ) from exc
    return FastMCP


def create_mcp_server(
    server: MemoryMCPServer,
):
    FastMCP = _load_fastmcp()
    mcp = FastMCP(
        _MCP_SERVER_NAME,
        json_response=True,
        streamable_http_path="/",
    )

    @mcp.tool()
    def remember_fact(
        content: str,
        importance: float = 0.5,
        category: str = "general",
        domain: str | None = None,
        confidence: float = 1.0,
        supersedes: str | None = None,
        related_ids: list[str] | None = None,
        has_visual: bool = False,
        modality: str = "text",
        media: dict[str, Any] | None = None,
        media_type: str | None = None,
        text_description: str | None = None,
        max_contradictions: int = DEFAULT_MAX_CONTRADICTIONS,
    ) -> RememberFactResponse | ToolErrorModel:
        """Store a persistent fact, preference, or durable piece of knowledge."""
        try:
            request = RememberFactRequest(
                content=content,
                importance=importance,
                category=category,
                domain=domain,
                confidence=confidence,
                supersedes=supersedes,
                related_ids=related_ids or [],
                has_visual=has_visual,
                modality=modality,
                media=MediaInputModel.model_validate(media) if media else None,
                media_type=media_type,
                text_description=text_description,
                max_contradictions=max_contradictions,
            )
            return server.remember_fact(request)
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    def remember_episode(
        session_id: str,
        text: str,
        turn_number: int | None = None,
        participants: list[str] | None = None,
        summary: str | None = None,
        emotional_profile: dict[str, float] | None = None,
        emotional_valence: float | None = None,
        importance: float = 0.5,
        modality: str = "text",
        media: dict[str, Any] | None = None,
        media_type: str | None = None,
        source_mime_type: str | None = None,
    ) -> RememberEpisodeResponse | ToolErrorModel:
        """Store a concrete event, interaction, or session-specific memory."""
        try:
            request = RememberEpisodeRequest(
                session_id=session_id,
                text=text,
                turn_number=turn_number,
                participants=participants or ["user", "agent"],
                summary=summary,
                emotional_profile=emotional_profile or {},
                emotional_valence=emotional_valence,
                importance=importance,
                modality=modality,
                media=MediaInputModel.model_validate(media) if media else None,
                media_type=media_type,
                source_mime_type=source_mime_type,
            )
            return server.remember_episode(request)
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    def remember_procedure(
        content: str,
        steps: list[str],
        preconditions: list[str] | None = None,
        importance: float = 0.5,
        modality: str = "text",
        media: dict[str, Any] | None = None,
        media_type: str | None = None,
        text_description: str | None = None,
    ) -> RememberProcedureResponse | ToolErrorModel:
        """Store a reusable workflow, skill, or step-by-step procedure."""
        try:
            request = RememberProcedureRequest(
                content=content,
                steps=steps,
                preconditions=preconditions or [],
                importance=importance,
                modality=modality,
                media=MediaInputModel.model_validate(media) if media else None,
                media_type=media_type,
                text_description=text_description,
            )
            return server.remember_procedure(request)
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    def resolve_contradiction(
        keep_id: str,
        supersede_id: str,
    ) -> ResolveContradictionResponse | ToolErrorModel:
        """Mark an older fact as superseded by the kept fact."""
        try:
            return server.resolve_contradiction(
                ResolveContradictionRequest(keep_id=keep_id, supersede_id=supersede_id)
            )
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    def record_procedure_outcome(
        record_id: str,
        success: bool,
    ) -> RecordProcedureOutcomeResponse | ToolErrorModel:
        """Record whether a stored procedure succeeded or failed when executed."""
        try:
            return server.record_procedure_outcome(
                RecordProcedureOutcomeRequest(record_id=record_id, success=success)
            )
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    def recall_memories(
        query: str,
        top_k: int = 5,
        memory_types: list[str] | None = None,
    ) -> RecallMemoriesResponse | ToolErrorModel:
        """Recall the most relevant memories for a natural-language query."""
        try:
            return server.recall_memories(
                RecallMemoriesRequest(query=query, top_k=top_k, memory_types=memory_types)
            )
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    def recall_procedures(
        task: str,
        top_k: int = 3,
    ) -> RecallProceduresResponse | ToolErrorModel:
        """Recall the best procedures for a task using similarity plus outcome ranking."""
        try:
            return server.recall_procedures(RecallProceduresRequest(task=task, top_k=top_k))
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    def recall_episodes(
        mode: str,
        limit: int = 5,
        session_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> RecallEpisodesResponse | ToolErrorModel:
        """Recall episodic memories by recent activity, session, or time range."""
        try:
            payload: dict[str, Any] = {"mode": mode}
            if mode == "recent":
                payload["limit"] = limit
            elif mode == "session":
                payload["session_id"] = session_id
            elif mode == "time_range":
                payload["start"] = start
                payload["end"] = end
            return server.recall_episodes(payload)
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    def get_memory(
        memory_type: str,
        record_id: str,
    ) -> GetMemoryResponse | ToolErrorModel:
        """Fetch one memory by explicit type and record id."""
        try:
            return server.get_memory(
                GetMemoryRequest(memory_type=memory_type, record_id=record_id)
            )
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    def get_memory_overview() -> MemoryOverviewModel | ToolErrorModel:
        """Get memory counts, recent sessions, and the latest lifecycle events."""
        try:
            return server.overview()
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    async def preview_forgetting_cycle(
        max_decisions: int = 50,
    ) -> ForgettingReportModel | ToolErrorModel:
        """Preview forgetting decisions without mutating stored memories."""
        try:
            return await server.preview_forgetting_cycle(
                ForgettingCycleRequest(max_decisions=max_decisions)
            )
        except Exception as exc:
            return _map_exception(exc).payload

    @mcp.tool()
    async def run_forgetting_cycle(
        max_decisions: int = 50,
    ) -> ForgettingReportModel | ToolErrorModel:
        """Run one forgetting cycle and return a bounded report of what happened."""
        try:
            return await server.run_forgetting_cycle(
                ForgettingCycleRequest(max_decisions=max_decisions)
            )
        except Exception as exc:
            return _map_exception(exc).payload

    return mcp


def create_http_app(server: MemoryMCPServer):
    try:
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route
    except ImportError as exc:  # pragma: no cover - import depends on optional dependency
        raise RuntimeError(
            "Starlette is required for streamable-http transport. "
            "Install the API or MCP dependencies before starting the HTTP server."
        ) from exc

    mcp = create_mcp_server(server)

    async def health(_request) -> JSONResponse:
        server.validate_startup()
        return JSONResponse(
            {
                "status": "ok",
                "transport": server.config.transport,
                "mcp_path": server.config.streamable_http_path,
            }
        )

    @asynccontextmanager
    async def lifespan(_app):
        server.validate_startup()
        async with mcp.session_manager.run():
            yield

    return Starlette(
        routes=[
            Route("/health", endpoint=health),
            Mount(server.config.streamable_http_path, app=mcp.streamable_http_app()),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    server = MemoryMCPServer(config=ServerConfig.from_env())
    server.validate_startup()

    if server.config.transport == "stdio":
        create_mcp_server(server).run(transport="stdio")
        return

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - import depends on optional dependency
        raise RuntimeError(
            "uvicorn is required for streamable-http transport."
        ) from exc

    app = create_http_app(server)
    uvicorn.run(app, host=server.config.host, port=server.config.port)


if __name__ == "__main__":
    main()
