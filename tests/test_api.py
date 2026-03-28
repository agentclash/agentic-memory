"""Verify the FastAPI boundary around the memory stores and retriever."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from api.app import create_app
from models.base import normalize_modality
from stores.episodic_store import EpisodicStoreError
from utils.embeddings import EmbeddingProviderError
from tests.helpers import HashingEmbedder


def make_client(*, media_root: str | None = None) -> httpx.AsyncClient:
    chroma_dir = tempfile.mkdtemp(prefix="memory_api_chroma_")
    app = create_app(
        chroma_path=chroma_dir,
        media_root=media_root,
        allowed_origins=["http://localhost:3000"],
        embedder=HashingEmbedder(dimensions=config.EMBEDDING_DIMENSIONS),
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
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
        await client.post(
            "/api/memories/procedural",
            json={
                "content": "Deploy retrieval with Docker",
                "steps": ["Build the image", "Run docker compose up"],
            },
        )

        response = await client.post("/api/retrieval/query", json={"query": "retrieval", "top_k": 3})
        data = response.json()

    assert response.status_code == 200
    assert {item["record"]["memory_type"] for item in data["results"]} == {
        "semantic",
        "episodic",
        "procedural",
    }


@pytest.mark.anyio
async def test_query_endpoint_filters_to_procedural_results():
    async with make_client() as client:
        await client.post("/api/memories/semantic", json={"content": "Semantic fact about Docker"})
        await client.post(
            "/api/memories/episodic/text",
            json={"session_id": "session-docker", "text": "We struggled with Docker networking"},
        )
        await client.post(
            "/api/memories/procedural",
            json={
                "content": "Deploy with Docker Compose",
                "steps": ["Build the image", "Run docker compose up"],
            },
        )

        response = await client.post(
            "/api/retrieval/query",
            json={"query": "Docker", "top_k": 3, "memory_types": ["procedural"]},
        )

    assert response.status_code == 200
    assert {item["record"]["memory_type"] for item in response.json()["results"]} == {"procedural"}


@pytest.mark.anyio
async def test_query_endpoint_rejects_invalid_memory_types():
    async with make_client() as client:
        response = await client.post(
            "/api/retrieval/query",
            json={"query": "Docker", "memory_types": ["procedural", "unknown"]},
        )

    assert response.status_code == 400
    assert "Unsupported memory_types" in response.json()["detail"]


@pytest.mark.anyio
async def test_query_endpoint_rejects_non_list_memory_types():
    async with make_client() as client:
        response = await client.post(
            "/api/retrieval/query",
            json={"query": "Docker", "memory_types": "procedural"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "memory_types must be an array of strings"


@pytest.mark.anyio
async def test_store_procedural_memory_and_record_outcome_via_api():
    async with make_client() as client:
        create = await client.post(
            "/api/memories/procedural",
            json={
                "content": "Deploy to Lambda",
                "steps": ["Package dependencies", "Run sam deploy"],
                "preconditions": ["AWS CLI configured"],
            },
        )
        created = create.json()["record"]
        update = await client.post(
            f"/api/memories/procedural/{created['id']}/outcome",
            json={"success": True},
        )

    assert create.status_code == 200
    assert created["steps"] == ["Package dependencies", "Run sam deploy"]
    assert created["preconditions"] == ["AWS CLI configured"]
    assert created["success_count"] == 0
    assert created["failure_count"] == 0
    assert created["wilson_score"] == 0.0
    assert update.status_code == 200
    assert update.json()["record"]["success_count"] == 1
    assert update.json()["record"]["failure_count"] == 0


@pytest.mark.anyio
async def test_best_procedures_endpoint_returns_ranked_results():
    async with make_client() as client:
        strong = await client.post(
            "/api/memories/procedural",
            json={
                "content": "Deploy to Lambda via SAM",
                "steps": ["Package dependencies", "Run sam deploy"],
            },
        )
        weaker = await client.post(
            "/api/memories/procedural",
            json={
                "content": "Deploy to Lambda via Serverless",
                "steps": ["Package dependencies", "Run serverless deploy"],
            },
        )
        weaker_id = weaker.json()["record"]["id"]
        for _ in range(9):
            await client.post(
                f"/api/memories/procedural/{weaker_id}/outcome",
                json={"success": True},
            )
        for _ in range(6):
            await client.post(
                f"/api/memories/procedural/{weaker_id}/outcome",
                json={"success": False},
            )
        strong_id = strong.json()["record"]["id"]
        for _ in range(9):
            await client.post(
                f"/api/memories/procedural/{strong_id}/outcome",
                json={"success": True},
            )
        response = await client.post(
            "/api/retrieval/best-procedures",
            json={"task": "deploy to Lambda", "top_k": 2},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 2
    assert payload["results"][0]["record"]["content"] == "Deploy to Lambda via SAM"
    assert payload["results"][0]["combined_score"] >= payload["results"][1]["combined_score"]
    assert "similarity" in payload["results"][0]
    assert "wilson_score" in payload["results"][0]


@pytest.mark.anyio
async def test_store_media_backed_procedure_via_api():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            response = await client.post(
                "/api/memories/procedural/file",
                data={
                    "content": "Review migration checklist",
                    "modality": "multimodal",
                    "media_type": "pdf",
                    "text_description": "Reference checklist for the migration sequence",
                },
                files=[
                    ("steps", (None, "Open the checklist")),
                    ("steps", (None, "Validate preconditions")),
                    ("steps", (None, "Run the migration")),
                    ("preconditions", (None, "Database backup completed")),
                    ("file", ("checklist.pdf", b"%PDF-1.4\nchecklist", "application/pdf")),
                ],
            )

        assert response.status_code == 200
        record = response.json()["record"]
        assert record["modality"] == "multimodal"
        assert record["media_type"] == "pdf"
        assert record["has_media"] is True
        assert record["media_ref"].endswith(os.path.join("documents", f"{record['id']}.pdf"))
        assert record["steps"] == [
            "Open the checklist",
            "Validate preconditions",
            "Run the migration",
        ]
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_procedural_api_validation_rejects_empty_steps_and_bad_modalities():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            empty_steps = await client.post(
                "/api/memories/procedural",
                json={"content": "Deploy to Lambda", "steps": []},
            )
            bad_file = await client.post(
                "/api/memories/procedural/file",
                data={"content": "Deploy to Lambda", "modality": "multimodal"},
                files=[
                    ("steps", (None, "Package dependencies")),
                    ("file", ("archive.tar.gz", b"bad-archive", "application/gzip")),
                ],
            )

        assert empty_steps.status_code == 400
        assert empty_steps.json()["detail"] == "steps must contain at least one entry"
        assert bad_file.status_code == 400
        assert bad_file.json()["detail"] == (
            "multimodal file uploads require a supported image, audio, video, or PDF file"
        )
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_text_query_finds_image_memory():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    fd, source_path = tempfile.mkstemp(suffix=".png", prefix="semantic_api_source_")
    os.close(fd)
    Path(source_path).write_bytes(b"diagram")
    try:
        async with make_client(media_root=media_root) as client:
            create = await client.post(
                "/api/memories/semantic",
                json={
                    "content": "image png architecture whiteboard",
                    "modality": "image",
                    "media_ref": source_path,
                    "media_type": "image",
                },
            )
            query = await client.post("/api/retrieval/query", json={"query": "architecture whiteboard", "top_k": 1})

        assert create.status_code == 200
        assert query.status_code == 200
        assert query.json()["results"][0]["record"]["modality"] == "image"
        print("  PASS  text retrieval can return image-backed memories")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)
        try:
            os.remove(source_path)
        except FileNotFoundError:
            pass


@pytest.mark.anyio
async def test_text_query_finds_audio_memory():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            create = await client.post(
                "/api/memories/episodic/file",
                data={"session_id": "session-audio", "content": "audio mpeg sprint recap"},
                files={"file": ("recap.mp3", b"sprint recap", "audio/mpeg")},
            )
            query = await client.post("/api/retrieval/query", json={"query": "sprint recap", "top_k": 1})

        assert create.status_code == 200
        assert query.status_code == 200
        assert query.json()["results"][0]["record"]["modality"] == "audio"
        print("  PASS  text retrieval can return audio-backed memories")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_store_semantic_memory_round_trips_media_contract():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    fd, source_path = tempfile.mkstemp(suffix=".png", prefix="semantic_api_source_")
    os.close(fd)
    Path(source_path).write_bytes(b"semantic-image")
    try:
        async with make_client(media_root=media_root) as client:
            create = await client.post(
                "/api/memories/semantic",
                json={
                    "content": "Architecture diagram for retrieval flow",
                    "modality": "image",
                    "media_ref": source_path,
                    "media_type": "image",
                    "text_description": "Whiteboard sketch of the retrieval stack",
                },
            )
            query = await client.post("/api/retrieval/query", json={"query": "architecture diagram", "top_k": 1})

        assert create.status_code == 200
        created = create.json()["record"]
        assert created["modality"] == "image"
        assert created["media_ref"] == os.path.join(media_root, "images", f"{created['id']}.png")
        assert created["media_type"] == "image"
        assert created["text_description"] == "Whiteboard sketch of the retrieval stack"
        assert created["has_media"] is True
        assert os.path.exists(created["media_ref"])

        assert query.status_code == 200
        record = query.json()["results"][0]["record"]
        assert record["modality"] == "image"
        assert record["media_ref"] == created["media_ref"]
        assert record["media_type"] == "image"
        assert record["text_description"] == "Whiteboard sketch of the retrieval stack"
        assert record["has_media"] is True
    finally:
        shutil.rmtree(media_root, ignore_errors=True)
        try:
            os.remove(source_path)
        except FileNotFoundError:
            pass


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
async def test_query_by_image_endpoint_returns_vector_metadata_and_text_match():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            await client.post(
                "/api/memories/semantic",
                json={"content": "image png architecture memory"},
            )
            response = await client.post(
                "/api/retrieval/query-by-image",
                data={"top_k": "1", "memory_types": "semantic"},
                files={"file": ("diagram.png", b"architecture", "image/png")},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["query_type"] == "vector"
        assert payload["source_modality"] == "image"
        assert payload["results"][0]["record"]["memory_type"] == "semantic"
        print("  PASS  image query endpoint embeds uploads and returns vector query metadata")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_query_by_audio_endpoint_returns_vector_metadata_and_text_match():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            await client.post(
                "/api/memories/semantic",
                json={"content": "audio mpeg sprint recap memory"},
            )
            response = await client.post(
                "/api/retrieval/query-by-audio",
                data={"top_k": "1", "memory_types": "semantic"},
                files={"file": ("recap.mp3", b"sprint recap", "audio/mpeg")},
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["query_type"] == "vector"
        assert payload["source_modality"] == "audio"
        assert payload["results"][0]["record"]["memory_type"] == "semantic"
        print("  PASS  audio query endpoint embeds uploads and returns vector query metadata")
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_query_by_image_rejects_non_image_upload_with_clear_message():
    async with make_client() as client:
        response = await client.post(
            "/api/retrieval/query-by-image",
            files={"file": ("notes.txt", b"not-an-image", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "query-by-image requires a supported image file upload"


@pytest.mark.anyio
async def test_store_file_episode_allows_non_pdf_multimodal_upload():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            response = await client.post(
                "/api/memories/episodic/file",
                data={"session_id": "session-multimodal", "modality": "multimodal"},
                files={"file": ("clip.mp3", b"fake-audio", "audio/mpeg")},
            )

        assert response.status_code == 200
        assert response.json()["record"]["modality"] == "multimodal"
        assert response.json()["record"]["media_type"] == "audio"
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_multimodal_file_upload_rejects_unsupported_file_type():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    try:
        async with make_client(media_root=media_root) as client:
            response = await client.post(
                "/api/memories/episodic/file",
                data={"session_id": "session-multimodal", "modality": "multimodal"},
                files={"file": ("archive.tar.gz", b"bad-archive", "application/gzip")},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "multimodal file uploads require a supported image, audio, video, or PDF file"
        )
    finally:
        shutil.rmtree(media_root, ignore_errors=True)


@pytest.mark.anyio
async def test_semantic_memory_requires_media_ref_for_non_text_modalities():
    async with make_client() as client:
        response = await client.post(
            "/api/memories/semantic",
            json={
                "content": "Bad semantic media contract",
                "modality": "image",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "media_ref is required when modality is not text"


@pytest.mark.anyio
async def test_semantic_memory_rejects_modality_media_type_mismatch():
    async with make_client() as client:
        response = await client.post(
            "/api/memories/semantic",
            json={
                "content": "Bad semantic modality contract",
                "modality": "image",
                "media_ref": "clip.mp3",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "media_type 'audio' does not match modality 'image'"


@pytest.mark.anyio
async def test_semantic_memory_rejects_invalid_related_ids():
    async with make_client() as client:
        response = await client.post(
            "/api/memories/semantic",
            json={
                "content": "Bad semantic relation payload",
                "related_ids": ["ok", 123],
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "related_ids must contain only strings"


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
async def test_failed_semantic_write_cleans_up_owned_media():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    fd, source_path = tempfile.mkstemp(suffix=".png", prefix="semantic_api_source_")
    os.close(fd)
    Path(source_path).write_bytes(b"semantic-image")
    try:
        async with make_client(media_root=media_root) as client:
            await client.get("/api/overview")
            service = client.app.state.service

            def fail_store(record):
                owned_ref = service.media_store.store(record.media_ref, record.id)
                record.media_ref = owned_ref
                raise RuntimeError("synthetic store failure")

            service.semantic_store.store = fail_store
            response = await client.post(
                "/api/memories/semantic",
                json={
                    "content": "This semantic write should fail",
                    "modality": "image",
                    "media_ref": source_path,
                    "media_type": "image",
                },
            )

        assert response.status_code == 500
        assert not any(path.is_file() for path in Path(media_root).rglob("*"))
    finally:
        shutil.rmtree(media_root, ignore_errors=True)
        try:
            os.remove(source_path)
        except FileNotFoundError:
            pass


@pytest.mark.anyio
async def test_bad_request_semantic_write_cleans_up_owned_media():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    fd, source_path = tempfile.mkstemp(suffix=".png", prefix="semantic_api_source_")
    os.close(fd)
    Path(source_path).write_bytes(b"semantic-image")
    try:
        async with make_client(media_root=media_root) as client:
            await client.get("/api/overview")
            service = client.app.state.service

            def fail_store(record):
                owned_ref = service.media_store.store(record.media_ref, record.id)
                record.media_ref = owned_ref
                raise ValueError("synthetic validation failure")

            service.semantic_store.store = fail_store
            response = await client.post(
                "/api/memories/semantic",
                json={
                    "content": "This semantic write should fail with a 400",
                    "modality": "image",
                    "media_ref": source_path,
                    "media_type": "image",
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "synthetic validation failure"
        assert not any(path.is_file() for path in Path(media_root).rglob("*"))
    finally:
        shutil.rmtree(media_root, ignore_errors=True)
        try:
            os.remove(source_path)
        except FileNotFoundError:
            pass


@pytest.mark.anyio
async def test_semantic_provider_failure_returns_502_and_cleans_up_owned_media():
    media_root = tempfile.mkdtemp(prefix="memory_api_media_")
    fd, source_path = tempfile.mkstemp(suffix=".png", prefix="semantic_api_source_")
    os.close(fd)
    Path(source_path).write_bytes(b"semantic-image")
    try:
        async with make_client(media_root=media_root) as client:
            await client.get("/api/overview")
            service = client.app.state.service

            def fail_store(record):
                owned_ref = service.media_store.store(record.media_ref, record.id)
                record.media_ref = owned_ref
                raise EmbeddingProviderError("Gemini embedding provider failed after retries")

            service.semantic_store.store = fail_store
            response = await client.post(
                "/api/memories/semantic",
                json={
                    "content": "This semantic write should fail with a provider error",
                    "modality": "image",
                    "media_ref": source_path,
                    "media_type": "image",
                },
            )

        assert response.status_code == 502
        assert response.json()["detail"] == "Gemini embedding provider failed after retries"
        assert not any(path.is_file() for path in Path(media_root).rglob("*"))
    finally:
        shutil.rmtree(media_root, ignore_errors=True)
        try:
            os.remove(source_path)
        except FileNotFoundError:
            pass


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
async def test_vector_query_events_include_source_modality():
    async with make_client() as client:
        await client.post("/api/memories/semantic", json={"content": "image png architecture memory"})
        query = await client.post(
            "/api/retrieval/query-by-image",
            data={"top_k": "1"},
            files={"file": ("diagram.png", b"architecture", "image/png")},
        )
        events = await client.get("/api/events", params={"limit": 10})

    assert query.status_code == 200
    ranked = next(event for event in events.json()["events"] if event["event_type"] == "memory.ranked")
    retrieved = next(event for event in events.json()["events"] if event["event_type"] == "memory.retrieved")
    assert ranked["data"]["query_type"] == "vector"
    assert ranked["data"]["query_metadata"]["source_modality"] == "image"
    assert retrieved["data"]["query_metadata"]["source_modality"] == "image"


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
