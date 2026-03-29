from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from forgetting.contradiction import ContradictionCandidate
from forgetting.service import ForgettingDecision, ForgettingReport
from mcp_server.schemas import (
    DEFAULT_MAX_CONTRADICTIONS,
    DEFAULT_MAX_DECISIONS,
    ForgettingReportModel,
    GetMemoryRequest,
    InlineContentModel,
    MCPToolError,
    MediaInputModel,
    RecallEpisodesRequestAdapter,
    RememberFactRequest,
    RememberProcedureRequest,
    serialise_forgetting_report,
    serialise_procedural_match,
    serialise_remember_fact_response,
    get_memory_by_type,
)
from models.procedural import ProceduralMemory
from models.semantic import SemanticMemory
from runtime import build_runtime
from stores.procedural_store import ProceduralMatch
from tests.helpers import HashingEmbedder, cleanup_dir, make_temp_chroma_dir


def test_recall_episodes_request_supports_all_locked_modes():
    recent = RecallEpisodesRequestAdapter.validate_python({"mode": "recent", "limit": 7})
    session = RecallEpisodesRequestAdapter.validate_python(
        {"mode": "session", "session_id": "session-123"}
    )
    time_range = RecallEpisodesRequestAdapter.validate_python(
        {
            "mode": "time_range",
            "start": "2026-03-29T00:00:00+00:00",
            "end": "2026-03-30T00:00:00+00:00",
        }
    )

    assert recent.mode == "recent"
    assert recent.limit == 7
    assert session.mode == "session"
    assert session.session_id == "session-123"
    assert time_range.mode == "time_range"
    assert time_range.start < time_range.end


def test_recall_episodes_request_rejects_invalid_time_ranges():
    with pytest.raises(ValidationError):
        RecallEpisodesRequestAdapter.validate_python(
            {
                "mode": "time_range",
                "start": "2026-03-30T00:00:00+00:00",
                "end": "2026-03-29T00:00:00+00:00",
            }
        )


def test_media_input_supports_file_path_or_inline_base64_only():
    file_input = MediaInputModel(file_path="/tmp/screenshot.png")
    inline_input = MediaInputModel(
        inline_content=InlineContentModel(
            inline_base64="ZmFrZQ==",
            mime_type="image/png",
            filename="screenshot.png",
        )
    )

    assert file_input.file_path == "/tmp/screenshot.png"
    assert inline_input.inline_content is not None

    with pytest.raises(ValidationError):
        MediaInputModel(
            file_path="/tmp/screenshot.png",
            inline_content=InlineContentModel(
                inline_base64="ZmFrZQ==",
                mime_type="image/png",
            ),
        )


def test_remember_fact_request_bounds_contradiction_count():
    request = RememberFactRequest(content="Embedding size is 768")
    assert request.max_contradictions == DEFAULT_MAX_CONTRADICTIONS

    with pytest.raises(ValidationError):
        RememberFactRequest(content="Embedding size is 768", max_contradictions=999)


def test_remember_procedure_request_rejects_blank_steps():
    with pytest.raises(ValidationError):
        RememberProcedureRequest(content="Deploy the app", steps=[""])


def test_serialise_remember_fact_response_reports_truncation():
    record = SemanticMemory(content="Embedding size is 768")
    candidates = [
        ContradictionCandidate(record=SemanticMemory(content=f"Candidate {index}"), similarity=0.9 - index * 0.1)
        for index in range(3)
    ]

    payload = serialise_remember_fact_response(
        record=record,
        contradiction_status="completed",
        contradiction_candidates=candidates,
        max_contradictions=1,
    )

    assert payload.record.content == "Embedding size is 768"
    assert payload.contradiction_check_order == "after_store"
    assert payload.total_contradictions == 3
    assert payload.contradictions_truncated is True
    assert len(payload.potential_contradictions) == 1
    assert payload.model_dump(mode="json", exclude_none=True)["contradiction_check"] == {
        "status": "completed"
    }


def test_serialise_forgetting_report_caps_decisions():
    report = ForgettingReport(
        dry_run=True,
        decisions=[
            ForgettingDecision(
                record_id=f"record-{index}",
                memory_type="semantic",
                action="keep",
                reason="time_decay",
                score=0.1 * index,
            )
            for index in range(DEFAULT_MAX_DECISIONS + 5)
        ],
    )

    payload = serialise_forgetting_report(report)

    assert payload.status == "completed"
    assert payload.displayed_decisions == DEFAULT_MAX_DECISIONS
    assert payload.total_decisions == DEFAULT_MAX_DECISIONS + 5
    assert payload.decisions_truncated is True
    assert len(payload.decisions) == DEFAULT_MAX_DECISIONS


def test_procedural_match_serializer_exposes_all_scores():
    record = ProceduralMemory(content="Deploy with Docker", steps=["docker compose up"])
    match = ProceduralMatch(
        record=record,
        similarity=0.75,
        wilson_score=0.55,
        combined_score=0.68,
    )

    payload = serialise_procedural_match(match)

    assert payload.similarity == 0.75
    assert payload.wilson_score == 0.55
    assert payload.combined_score == 0.68


def test_get_memory_by_type_routes_to_the_correct_store():
    chroma_dir = make_temp_chroma_dir("mcp_schemas_")
    runtime = build_runtime(
        chroma_path=chroma_dir,
        embedder=HashingEmbedder(dimensions=64),
        embedding_dimensions=64,
    )
    try:
        semantic = SemanticMemory(content="Chroma path is isolated per runtime")
        procedural = ProceduralMemory(
            content="Deploy the app",
            steps=["Build the image", "Start the service"],
        )
        runtime.semantic_store.store(semantic)
        runtime.procedural_store.store(procedural)

        semantic_result = get_memory_by_type(
            runtime,
            GetMemoryRequest(memory_type="semantic", record_id=semantic.id),
        )
        procedural_result = get_memory_by_type(
            runtime,
            GetMemoryRequest(memory_type="procedural", record_id=procedural.id),
        )

        assert semantic_result is not None
        assert semantic_result.id == semantic.id
        assert procedural_result is not None
        assert procedural_result.id == procedural.id
    finally:
        cleanup_dir(chroma_dir)


def test_typed_tool_errors_produce_stable_payloads():
    error = MCPToolError(
        code="dependency_unavailable",
        message="Embedding provider timed out",
        retryable=True,
        details={"provider": "gemini"},
    )

    assert error.payload.code == "dependency_unavailable"
    assert error.payload.retryable is True
    assert error.payload.details == {"provider": "gemini"}


def test_already_running_forgetting_payload_is_model_shaped():
    payload = ForgettingReportModel(
        status="already_running",
        dry_run=True,
    ).model_dump(mode="json", exclude_none=True)

    assert payload == {
        "status": "already_running",
        "dry_run": True,
        "decisions": [],
        "displayed_decisions": 0,
        "total_decisions": 0,
        "decisions_truncated": False,
    }
