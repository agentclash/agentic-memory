from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest

from mcp_server.server import MemoryMCPServer, ServerConfig
from mcp_server.schemas import (
    ForgettingCycleRequest,
    GetMemoryRequest,
    RememberEpisodeRequest,
    RememberFactRequest,
)
from tests.helpers import HashingEmbedder, cleanup_dir, make_temp_chroma_dir


def make_server(
    *,
    transport: str = "stdio",
    chroma_path: str | None = None,
    media_root: str | None = None,
) -> MemoryMCPServer:
    resolved_chroma = chroma_path or make_temp_chroma_dir("mcp_server_")
    resolved_media_root = media_root or make_temp_chroma_dir("mcp_media_")
    return MemoryMCPServer(
        config=ServerConfig(
            transport=transport,
            chroma_path=resolved_chroma,
            media_root=resolved_media_root,
            embedding_dimensions=32,
        ),
        embedder=HashingEmbedder(dimensions=32),
    )


def test_server_remember_fact_round_trip():
    server = make_server()
    try:
        response = server.remember_fact(
            RememberFactRequest(
                content="The MCP server should reuse one runtime per process",
                category="architecture",
            )
        )

        assert response.record.content == "The MCP server should reuse one runtime per process"
        assert response.contradiction_check.status == "completed"

        stored = server.get_memory(
            GetMemoryRequest(memory_type="semantic", record_id=response.record.id)
        )
        assert stored.record is not None
        assert stored.record.id == response.record.id
    finally:
        cleanup_dir(server.config.chroma_path or "")
        cleanup_dir(server.config.media_root or "")


def test_server_materializes_inline_media_for_episode_storage():
    server = make_server()
    try:
        payload = base64.b64encode(b"diagram contents for multimodal recall").decode("ascii")
        response = server.remember_episode(
            RememberEpisodeRequest(
                session_id="session-1",
                text="Architecture diagram from planning session",
                media={
                    "inline_content": {
                        "inline_base64": payload,
                        "mime_type": "image/png",
                        "filename": "diagram.png",
                    }
                },
            )
        )

        assert response.record.media_ref is not None
        assert Path(response.record.media_ref).exists()
        assert response.record.modality == "image"
    finally:
        cleanup_dir(server.config.chroma_path or "")
        cleanup_dir(server.config.media_root or "")


def test_prepare_media_cleans_up_temp_file_when_media_type_resolution_fails():
    server = make_server()
    before = {path for path in Path("/tmp").glob("agentic_memory_mcp_*")}
    try:
        payload = base64.b64encode(b"not a supported media type").decode("ascii")
        with pytest.raises(ValueError):
            server._prepare_media(  # type: ignore[attr-defined]
                RememberEpisodeRequest(
                    session_id="session-1",
                    text="Unsupported inline media",
                    media={
                        "inline_content": {
                            "inline_base64": payload,
                            "mime_type": "application/octet-stream",
                            "filename": "blob.bin",
                        }
                    },
                ).media,
                None,
            )
    finally:
        after = {path for path in Path("/tmp").glob("agentic_memory_mcp_*")}
        cleanup_dir(server.config.chroma_path or "")
        cleanup_dir(server.config.media_root or "")

    assert after == before


@pytest.mark.anyio
async def test_server_forgetting_cycle_is_single_flight():
    server = make_server()
    try:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = asyncio.Event()

        original = server.runtime().forgetting_service.run_cycle

        def slow_run_cycle(*, dry_run: bool):
            started.set()
            future = asyncio.run_coroutine_threadsafe(release.wait(), loop)
            future.result(timeout=2)
            return original(dry_run=dry_run)

        server.runtime().forgetting_service.run_cycle = slow_run_cycle  # type: ignore[method-assign]
        try:
            first_task = asyncio.create_task(
                server.preview_forgetting_cycle(ForgettingCycleRequest(max_decisions=5))
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            second = await server.preview_forgetting_cycle(ForgettingCycleRequest(max_decisions=5))
            release.set()
            first = await asyncio.wait_for(first_task, timeout=1)
        finally:
            server.runtime().forgetting_service.run_cycle = original  # type: ignore[method-assign]

        assert first.status == "completed"
        assert second.status == "already_running"
    finally:
        cleanup_dir(server.config.chroma_path or "")
        cleanup_dir(server.config.media_root or "")


def test_streamable_http_config_allows_relative_paths_but_keeps_transport_valid():
    config = ServerConfig(
        transport="streamable-http",
        chroma_path="./chroma_db",
        media_root="./data/media",
    ).validate()

    assert config.transport == "streamable-http"


def test_invalid_transport_is_rejected():
    with pytest.raises(ValueError):
        ServerConfig(transport="http2").validate()


def test_invalid_port_is_rejected():
    with pytest.raises(ValueError):
        ServerConfig(port=0).validate()
