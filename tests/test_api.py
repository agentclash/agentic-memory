"""Verify the FastAPI boundary around the memory stores and retriever."""

import os
import sys
import tempfile

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import create_app
from tests.helpers import HashingEmbedder


def make_client() -> httpx.AsyncClient:
    chroma_dir = tempfile.mkdtemp(prefix="memory_api_chroma_")
    upload_dir = tempfile.mkdtemp(prefix="memory_api_uploads_")
    app = create_app(
        chroma_path=chroma_dir,
        upload_dir=upload_dir,
        allowed_origins=["http://localhost:3000"],
        embedder=HashingEmbedder(),
    )
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


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
async def test_store_file_episode_and_temporal_queries():
    async with make_client() as client:
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
    assert record["source_mime_type"] == "image/png"
    assert record["media_ref"].endswith(".png")
    assert recent.status_code == 200
    assert session.status_code == 200
    assert time_range.status_code == 200
    assert recent.json()["records"][0]["memory_type"] == "episodic"
    assert session.json()["records"][0]["session_id"] == "session-media"
    assert time_range.json()["records"][0]["modality"] == "image"


@pytest.mark.anyio
async def test_store_file_episode_inferrs_modality_from_extension_and_mime():
    async with make_client() as client:
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
    assert pdf.json()["record"]["modality"] == "pdf"


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
