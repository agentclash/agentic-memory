"""Verify the FastAPI boundary around the memory stores and retriever."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import create_app
from models.base import normalize_modality
from stores.episodic_store import EpisodicStoreError
from tests.helpers import HashingEmbedder


def make_client(*, media_root: str | None = None) -> httpx.AsyncClient:
    chroma_dir = tempfile.mkdtemp(prefix="memory_api_chroma_")
    app = create_app(
        chroma_path=chroma_dir,
        media_root=media_root,
        allowed_origins=["http://localhost:3000"],
        embedder=HashingEmbedder(),
    )
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    client.app = app
    return client


@pytest.mark.anyio
async def test_store_semantic_and_query_mixed_results():
    async with make_client() as client:
        await client.post("/api/memories/semantic", json={"content": "Semantic fact about retrieval"})
        await client.post(
            "/api/memories/episodic/text",
            json={"session_id": "session-a", "text": "We debugged retrieval during a session"},
        )

        response = await client.post("/api/retrieval/query", json={"query": "retrieval", "top_k": 2})
        data = response.json()

    assert response.status_code == 200
    assert {item["record"]["memory_type"] for item in data["results"]} == {"semantic", "episodic"}


@pytest.mark.anyio
async def test_store_semantic_memory_round_trips_media_contract():
    async with make_client() as client:
        create = await client.post(
            "/api/memories/semantic",
            json={
                "content": "Architecture diagram for retrieval flow",
                "modality": "image",
                "media_ref": "/tmp/diagram.png",
                "media_type": "image",
                "text_description": "Whiteboard sketch of the retrieval stack",
            },
        )
        query = await client.post("/api/retrieval/query", json={"query": "architecture diagram", "top_k": 1})

    assert create.status_code == 200
    created = create.json()["record"]
    assert created["modality"] == "image"
    assert created["media_ref"] == "/tmp/diagram.png"
    assert created["media_type"] == "image"
    assert created["text_description"] == "Whiteboard sketch of the retrieval stack"
    assert created["has_media"] is True

    assert query.status_code == 200
    record = query.json()["results"][0]["record"]
    assert record["modality"] == "image"
    assert record["media_ref"] == "/tmp/diagram.png"
    assert record["media_type"] == "image"
    assert record["text_description"] == "Whiteboard sketch of the retrieval stack"
    assert record["has_media"] is True


@pytest.mark.anyio
async def test_semantic_memory_rejects_invalid_media_type():
    async with make_client() as client:
        response = await client.post(
            "/api/memories/semantic",
            json={
                "content": "Bad semantic media contract",
                "media_type": "archive",
            },
        )

    assert response.status_code == 400
    assert "Unsupported media_type" in response.json()["detail"]


@pytest.mark.anyio
async def test_semantic_memory_rejects_non_string_modality():
    async with make_client() as client:
        response = await client.post(
            "/api/memories/semantic",
            json={
                "content": "Bad semantic modality",
                "modality": True,
            },
        )

    assert response.status_code == 400
    assert "Unsupported modality type" in response.json()["detail"]


@pytest.mark.anyio
async def test_store_file_episode_and_temporal_queries():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            response = await client.post(
                "/api/memories/episodic/file",
                data={
                    "session_id": "session-media",
                    "content": "Screenshot of a failed run",
                },
                files={"file": ("failure.png", b"fake-image", "image/png")},
            )
            record = response.json()["record"]

            recent = await client.get("/api/episodes/recent", params={"n": 1})
            session = await client.get("/api/episodes/session/session-media")
            time_range = await client.get(
                "/api/episodes/time-range",
                params={
                    "start": "2026-01-01T00:00:00+00:00",
                    "end": "2030-01-01T00:00:00+00:00",
                },
            )

        assert response.status_code == 200
        assert record["modality"] == "image"
        assert record["media_type"] == "image"
        assert record["source_mime_type"] == "image/png"
        assert record["has_media"] is True
        assert record["media_ref"] == os.path.join(media_root, "images", f"{record['id']}.png")
        assert os.path.exists(record["media_ref"])
        assert recent.status_code == 200
        assert session.status_code == 200
        assert time_range.status_code == 200
        assert recent.json()["records"][0]["memory_type"] == "episodic"
        assert session.json()["records"][0]["session_id"] == "session-media"
        assert time_range.json()["records"][0]["modality"] == "image"
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_store_file_episode_infers_modality_from_extension_and_mime():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            audio = await client.post(
                "/api/memories/episodic/file",
                data={"session_id": "session-audio"},
                files={"file": ("clip.mp3", b"fake-audio", "audio/mpeg")},
            )
            pdf = await client.post(
                "/api/memories/episodic/file",
                data={"session_id": "session-pdf"},
                files={"file": ("notes.pdf", b"%PDF-1.4\n%", "application/pdf")},
            )

        assert audio.status_code == 200
        assert pdf.status_code == 200
        assert audio.json()["record"]["modality"] == "audio"
        assert audio.json()["record"]["media_type"] == "audio"
        assert pdf.json()["record"]["modality"] == "multimodal"
        assert pdf.json()["record"]["media_type"] == "pdf"
        assert pdf.json()["record"]["source_mime_type"] == "application/pdf"
        assert audio.json()["record"]["media_ref"].endswith(
            os.path.join("audio", f"{audio.json()['record']['id']}.mp3")
        )
        assert pdf.json()["record"]["media_ref"].endswith(
            os.path.join("documents", f"{pdf.json()['record']['id']}.pdf")
        )
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_store_file_episode_rejects_non_pdf_multimodal_upload():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            response = await client.post(
                "/api/memories/episodic/file",
                data={"session_id": "session-multimodal", "modality": "multimodal"},
                files={"file": ("clip.mp3", b"fake-audio", "audio/mpeg")},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "multimodal file uploads currently require a PDF file"
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_text_episode_round_trips_emotional_profile_via_api():
    async with make_client() as client:
        create = await client.post(
            "/api/memories/episodic/text",
            json={
                "session_id": "session-emotions",
                "text": "We wrapped up the debugging session",
                "emotional_profile": {"relief": 0.9, "confidence": 0.6},
            },
        )
        recent = await client.get("/api/episodes/recent", params={"n": 1})

    assert create.status_code == 200
    assert create.json()["record"]["emotional_profile"] == {"relief": 0.9, "confidence": 0.6}
    assert recent.status_code == 200
    assert recent.json()["records"][0]["emotional_profile"] == {"relief": 0.9, "confidence": 0.6}


@pytest.mark.anyio
async def test_text_episode_rejects_invalid_emotional_profile_values():
    async with make_client() as client:
        response = await client.post(
            "/api/memories/episodic/text",
            json={
                "session_id": "session-invalid-emotions",
                "text": "This payload should fail",
                "emotional_profile": {"evil": "not-a-float"},
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "emotional_profile values must be numeric"


@pytest.mark.anyio
async def test_text_episode_treats_null_emotional_profile_as_empty():
    async with make_client() as client:
        response = await client.post(
            "/api/memories/episodic/text",
            json={
                "session_id": "session-null-emotions",
                "text": "Null emotional profile is acceptable",
                "emotional_profile": None,
            },
        )

    assert response.status_code == 200
    assert response.json()["record"]["emotional_profile"] == {}


def test_normalize_modality_maps_legacy_pdf_alias():
    assert normalize_modality("pdf") == "multimodal"


@pytest.mark.anyio
async def test_failed_file_episode_write_cleans_up_owned_media():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            await client.get("/api/overview")
            service = client.app.state.service

            def fail_store(record):
                raise EpisodicStoreError("synthetic store failure")

            service.episodic_store.store = fail_store
            response = await client.post(
                "/api/memories/episodic/file",
                data={"session_id": "session-fail"},
                files={"file": ("failure.png", b"fake-image", "image/png")},
            )

        assert response.status_code == 422
        assert not any(path.is_file() for path in Path(media_root).rglob("*"))
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_events_and_overview_reflect_activity():
    async with make_client() as client:
        await client.post("/api/memories/semantic", json={"content": "Overview fact"})
        await client.post(
            "/api/memories/episodic/text",
            json={"session_id": "session-events", "text": "Episode for the event stream"},
        )
        await client.get("/api/episodes/recent", params={"n": 1})

        overview = await client.get("/api/overview")
        events = await client.get("/api/events", params={"limit": 10})

    assert overview.status_code == 200
    assert events.status_code == 200
    assert overview.json()["semantic_count"] == 1
    assert overview.json()["episodic_count"] == 1
    assert any(event["event_type"] == "memory.accessed" for event in events.json()["events"])


@pytest.mark.anyio
async def test_events_and_overview_serialise_frozen_ranked_payloads():
    async with make_client() as client:
        await client.post("/api/memories/semantic", json={"content": "Python uses indentation"})
        await client.post(
            "/api/memories/episodic/text",
            json={"session_id": "session-ranked", "text": "We talked about Python retrieval behavior"},
        )
        query = await client.post("/api/retrieval/query", json={"query": "Python", "top_k": 2})
        overview = await client.get("/api/overview")
        events = await client.get("/api/events", params={"limit": 10})

    assert query.status_code == 200
    assert overview.status_code == 200
    assert events.status_code == 200
    ranked = next(event for event in events.json()["events"] if event["event_type"] == "memory.ranked")
    assert isinstance(ranked["data"]["results"], list)
    assert isinstance(ranked["data"]["weights"], dict)
