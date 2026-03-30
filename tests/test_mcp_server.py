from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from pathlib import Path

import pytest

from mcp_server.server import MemoryMCPServer, ServerConfig
from mcp_server.schemas import (
    ForgettingCycleRequest,
    GetMemoryRequest,
    RecallMemoriesRequest,
    RememberEpisodeRequest,
    RememberFactRequest,
)
from stores.episodic_store import MediaTooLargeError
from tests.helpers import HashingEmbedder, cleanup_dir, make_temp_chroma_dir


def make_server(
    *,
    transport: str = "stdio",
    chroma_path: str | None = None,
    media_root: str | None = None,
    max_media_bytes: int | None = None,
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
        max_media_bytes=max_media_bytes,
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
        assert response.record.media_ref.startswith(server.config.media_root or "")
        assert "agentic_memory_mcp_" not in response.record.media_ref
    finally:
        cleanup_dir(server.config.chroma_path or "")
        cleanup_dir(server.config.media_root or "")


@pytest.mark.parametrize(
    ("suffix", "mime_type", "modality", "media_type", "content", "query", "directory"),
    [
        (".png", "image/png", "image", "image", "UI screenshot from deploy review", "deploy review", "images"),
        (".mp3", "audio/mpeg", "audio", "audio", "Sprint recap audio from standup", "sprint recap", "audio"),
        (".mp4", "video/mp4", "video", "video", "Release demo video for onboarding", "release demo", "video"),
        (
            ".pdf",
            "application/pdf",
            "multimodal",
            "pdf",
            "Migration checklist PDF for launch",
            "migration checklist",
            "documents",
        ),
    ],
)
def test_server_stores_local_media_paths_and_text_recall_finds_them(
    suffix: str,
    mime_type: str,
    modality: str,
    media_type: str,
    content: str,
    query: str,
    directory: str,
):
    server = make_server()
    fd, source_path = tempfile.mkstemp(suffix=suffix, prefix="mcp_media_source_")
    os.close(fd)
    Path(source_path).write_bytes(content.encode("utf-8"))
    try:
        response = server.remember_episode(
            RememberEpisodeRequest(
                session_id="session-media",
                text=content,
                modality=modality,
                media_type=media_type,
                media={"file_path": source_path},
                source_mime_type=mime_type,
            )
        )
        recall = server.recall_memories(
            RecallMemoriesRequest(query=query, top_k=1, memory_types=["episodic"])
        )

        assert response.record.media_ref is not None
        assert Path(response.record.media_ref).exists()
        assert response.record.media_ref != source_path
        assert response.record.media_ref.endswith(
            os.path.join(directory, f"{response.record.id}{suffix}")
        )
        assert recall.results[0].record.id == response.record.id
        assert recall.results[0].record.media_type == media_type
    finally:
        cleanup_dir(server.config.chroma_path or "")
        cleanup_dir(server.config.media_root or "")
        Path(source_path).unlink(missing_ok=True)


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


def test_prepare_media_rejects_oversized_inline_payloads_before_tempfile_write():
    server = make_server(max_media_bytes=8)
    before = {path for path in Path("/tmp").glob("agentic_memory_mcp_*")}
    try:
        payload = base64.b64encode(b"0123456789").decode("ascii")
        with pytest.raises(MediaTooLargeError) as exc_info:
            server._prepare_media(  # type: ignore[attr-defined]
                RememberEpisodeRequest(
                    session_id="session-1",
                    text="Oversized screenshot",
                    media={
                        "inline_content": {
                            "inline_base64": payload,
                            "mime_type": "image/png",
                            "filename": "oversized.png",
                        }
                    },
                ).media,
                None,
            )
        assert "size_bytes=10" in str(exc_info.value)
        assert "limit_bytes=8" in str(exc_info.value)
    finally:
        after = {path for path in Path("/tmp").glob("agentic_memory_mcp_*")}
        cleanup_dir(server.config.chroma_path or "")
        cleanup_dir(server.config.media_root or "")

    assert after == before


def test_prepare_media_reports_local_path_errors_from_server_host_view():
    server = make_server()
    missing_path = "/tmp/agentic_memory_missing_screenshot.png"
    try:
        with pytest.raises(FileNotFoundError) as exc_info:
            server._prepare_media(  # type: ignore[attr-defined]
                RememberEpisodeRequest(
                    session_id="session-1",
                    text="Missing screenshot",
                    media={"file_path": missing_path},
                ).media,
                "image",
            )
        assert "MCP server host" in str(exc_info.value)
        assert missing_path in str(exc_info.value)
    finally:
        cleanup_dir(server.config.chroma_path or "")
        cleanup_dir(server.config.media_root or "")


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
