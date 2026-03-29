"""Edge-case API tests around the shared runtime refactor."""

import os
import sys
import tempfile

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import create_app
from tests.helpers import HashingEmbedder


def make_client(*, embedding_dimensions: int = 8) -> httpx.AsyncClient:
    chroma_dir = tempfile.mkdtemp(prefix="memory_api_runtime_edges_")
    app = create_app(
        chroma_path=chroma_dir,
        allowed_origins=["http://localhost:3000"],
        embedder=HashingEmbedder(dimensions=embedding_dimensions),
        embedding_dimensions=embedding_dimensions,
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    client.app = app
    return client


@pytest.mark.anyio
async def test_semantic_write_swallows_valueerror_from_contradiction_lookup():
    async with make_client() as client:
        service = client.app.state.service
        assert service is None

        await client.get("/api/overview")
        client.app.state.service.contradiction_detector.find_potential_contradictions = (
            lambda record: (_ for _ in ()).throw(ValueError("synthetic contradiction miss"))
        )

        response = await client.post(
            "/api/memories/semantic",
            json={"content": "ValueError contradiction lookup should be tolerated"},
        )

    assert response.status_code == 200
    assert response.json()["potential_contradictions"] == []


@pytest.mark.anyio
async def test_semantic_write_surfaces_programming_error_from_contradiction_lookup():
    async with make_client() as client:
        await client.get("/api/overview")
        client.app.state.service.contradiction_detector.find_potential_contradictions = (
            lambda record: (_ for _ in ()).throw(TypeError("synthetic programming error"))
        )

        response = await client.post(
            "/api/memories/semantic",
            json={"content": "TypeError contradiction lookup should not be suppressed"},
        )

    assert response.status_code == 500


@pytest.mark.anyio
async def test_create_app_uses_runtime_embedding_dimensions_for_service_retriever():
    async with make_client(embedding_dimensions=6) as client:
        await client.get("/api/overview")

        try:
            client.app.state.service.retriever.query_by_vector([0.0] * 5, top_k=1)
            raise AssertionError("Expected retriever dimension mismatch")
        except ValueError as exc:
            assert "Expected query vector dimension 6" in str(exc)
