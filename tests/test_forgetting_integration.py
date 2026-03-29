"""Verify forgetting endpoints, contradiction lookup, and full cycle integration."""

import os
import shutil
import sys
import tempfile

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from api.app import create_app
from tests.helpers import HashingEmbedder


def make_client(*, media_root: str | None = None) -> httpx.AsyncClient:
    chroma_dir = tempfile.mkdtemp(prefix="memory_forgetting_chroma_")
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
async def test_semantic_store_returns_potential_contradictions():
    async with make_client() as client:
        r1 = await client.post(
            "/api/memories/semantic",
            json={"content": "The service runs on port 8080"},
        )
        assert r1.status_code == 200
        body1 = r1.json()
        assert "potential_contradictions" in body1
        assert isinstance(body1["potential_contradictions"], list)

        r2 = await client.post(
            "/api/memories/semantic",
            json={"content": "The service runs on port 9090"},
        )
        assert r2.status_code == 200
        body2 = r2.json()
        assert "potential_contradictions" in body2


@pytest.mark.anyio
async def test_forgetting_preview_returns_stable_report():
    async with make_client() as client:
        await client.post("/api/memories/semantic", json={"content": "Fact alpha"})
        await client.post("/api/memories/semantic", json={"content": "Fact beta"})

        response = await client.post("/api/forgetting/preview")
        assert response.status_code == 200
        report = response.json()

        assert report["dry_run"] is True
        assert report["scanned"] == 2
        assert isinstance(report["decisions"], list)
        assert len(report["decisions"]) == 2
        assert report["kept"] + report["faded"] + report["pruned"] == report["scanned"]

        for decision in report["decisions"]:
            assert "record_id" in decision
            assert "memory_type" in decision
            assert "action" in decision
            assert "score" in decision


@pytest.mark.anyio
async def test_forgetting_preview_does_not_mutate():
    async with make_client() as client:
        r = await client.post("/api/memories/semantic", json={"content": "Immutable fact"})
        record_id = r.json()["record"]["id"]

        await client.post("/api/forgetting/preview")

        overview = await client.get("/api/overview")
        assert overview.json()["semantic_count"] == 1


@pytest.mark.anyio
async def test_forgetting_run_executes_cycle():
    async with make_client() as client:
        await client.post("/api/memories/semantic", json={"content": "Will survive"})

        response = await client.post("/api/forgetting/run")
        assert response.status_code == 200
        report = response.json()

        assert report["dry_run"] is False
        assert report["scanned"] >= 1
        assert isinstance(report["decisions"], list)


@pytest.mark.anyio
async def test_resolve_endpoint_performs_supersession():
    async with make_client() as client:
        r_old = await client.post(
            "/api/memories/semantic",
            json={"content": "Old deployment region: us-east-1"},
        )
        old_id = r_old.json()["record"]["id"]

        r_new = await client.post(
            "/api/memories/semantic",
            json={"content": "New deployment region: eu-west-1"},
        )
        new_id = r_new.json()["record"]["id"]

        response = await client.post(
            "/api/forgetting/resolve",
            json={"keep_id": new_id, "supersede_id": old_id},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["superseded_id"] == old_id
        assert result["kept_id"] == new_id
        assert result["status"] == "resolved"


@pytest.mark.anyio
async def test_resolve_endpoint_rejects_missing_ids():
    async with make_client() as client:
        response = await client.post(
            "/api/forgetting/resolve",
            json={"keep_id": "nonexistent", "supersede_id": "also-nonexistent"},
        )
        assert response.status_code == 404


@pytest.mark.anyio
async def test_resolve_endpoint_rejects_empty_payload():
    async with make_client() as client:
        response = await client.post("/api/forgetting/resolve", json={})
        assert response.status_code == 400


@pytest.mark.anyio
async def test_forgetting_events_appear_in_event_stream():
    async with make_client() as client:
        await client.post("/api/memories/semantic", json={"content": "Event witness"})
        await client.post("/api/forgetting/preview")

        events_r = await client.get("/api/events?limit=50")
        event_types = [e["event_type"] for e in events_r.json()["events"]]
        assert "forgetting.cycle_dry_run" in event_types


@pytest.mark.anyio
async def test_full_cycle_supersede_then_prune():
    """End-to-end: store two facts, resolve supersession, run cycle, verify pruned."""
    async with make_client() as client:
        r_old = await client.post(
            "/api/memories/semantic",
            json={"content": "The API key rotates monthly", "importance": 0.3},
        )
        old_id = r_old.json()["record"]["id"]

        r_new = await client.post(
            "/api/memories/semantic",
            json={"content": "The API key rotates weekly", "importance": 0.8},
        )
        new_id = r_new.json()["record"]["id"]

        await client.post(
            "/api/forgetting/resolve",
            json={"keep_id": new_id, "supersede_id": old_id},
        )

        run_r = await client.post("/api/forgetting/run")
        report = run_r.json()
        decisions_by_id = {d["record_id"]: d for d in report["decisions"]}

        assert decisions_by_id[old_id]["action"] == "prune"
        assert decisions_by_id[old_id]["reason"] == "superseded"
        assert decisions_by_id[new_id]["action"] == "keep"

        overview = await client.get("/api/overview")
        assert overview.json()["semantic_count"] == 1
