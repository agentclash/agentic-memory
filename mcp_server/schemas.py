from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from forgetting.contradiction import ContradictionCandidate
from forgetting.service import ForgettingReport
from models.base import MemoryRecord
from models.episodic import EpisodicMemory
from models.procedural import ProceduralMemory
from models.semantic import SemanticMemory
from retrieval.ranking import RankedResult
from runtime import MemoryRuntime
from stores.procedural_store import ProceduralMatch

MemoryType = Literal["semantic", "episodic", "procedural"]
ForgettingAction = Literal["keep", "fade", "prune"]
ContradictionStatus = Literal["completed", "skipped", "error"]

DEFAULT_RETRIEVAL_TOP_K = 5
MAX_RETRIEVAL_TOP_K = 20
DEFAULT_RECENT_EPISODES = 5
MAX_RECENT_EPISODES = 50
DEFAULT_MAX_DECISIONS = 50
MAX_MAX_DECISIONS = 100
DEFAULT_MAX_CONTRADICTIONS = 5
MAX_MAX_CONTRADICTIONS = 10


class MCPModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolErrorModel(MCPModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None


class MCPToolError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.payload = ToolErrorModel(
            code=code,
            message=message,
            retryable=retryable,
            details=details,
        )


class InlineContentModel(MCPModel):
    inline_base64: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    filename: str | None = None


class MediaInputModel(MCPModel):
    file_path: str | None = None
    inline_content: InlineContentModel | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "MediaInputModel":
        if self.file_path and self.inline_content:
            raise ValueError("Provide either file_path or inline_content, not both")
        if not self.file_path and self.inline_content is None:
            raise ValueError("A media input requires file_path or inline_content")
        return self


class MemoryRecordModel(MCPModel):
    id: str
    memory_type: MemoryType
    modality: str
    content: str
    created_at: datetime
    last_accessed_at: datetime | None = None
    access_count: int
    importance: float
    media_ref: str | None = None
    media_type: str | None = None
    text_description: str | None = None
    has_media: bool
    category: str | None = None
    domain: str | None = None
    confidence: float | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    related_ids: list[str] | None = None
    has_visual: bool | None = None
    session_id: str | None = None
    turn_number: int | None = None
    participants: list[str] | None = None
    summary: str | None = None
    emotional_valence: float | None = None
    emotional_profile: dict[str, float] | None = None
    source_mime_type: str | None = None
    steps: list[str] | None = None
    preconditions: list[str] | None = None
    success_count: int | None = None
    failure_count: int | None = None
    total_outcomes: int | None = None
    success_rate: float | None = None
    wilson_score: float | None = None


class RankedMemoryResultModel(MCPModel):
    record: MemoryRecordModel
    raw_similarity: float
    recency_score: float
    importance_score: float
    final_score: float


class ProcedureMatchModel(MCPModel):
    record: MemoryRecordModel
    similarity: float
    wilson_score: float
    combined_score: float


class ContradictionCandidateModel(MCPModel):
    record: MemoryRecordModel
    similarity: float


class ContradictionCheckModel(MCPModel):
    status: ContradictionStatus
    detail: str | None = None


class RememberFactRequest(MCPModel):
    content: str = Field(min_length=1)
    importance: float = Field(default=0.5)
    category: str = Field(default="general", min_length=1)
    domain: str | None = None
    confidence: float = Field(default=1.0)
    supersedes: str | None = None
    related_ids: list[str] = Field(default_factory=list)
    has_visual: bool = False
    modality: str = "text"
    media: MediaInputModel | None = None
    media_type: Literal["image", "audio", "video", "pdf"] | None = None
    text_description: str | None = None
    max_contradictions: int = Field(
        default=DEFAULT_MAX_CONTRADICTIONS,
        ge=1,
        le=MAX_MAX_CONTRADICTIONS,
    )


class RememberEpisodeRequest(MCPModel):
    session_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    turn_number: int | None = None
    participants: list[str] = Field(default_factory=lambda: ["user", "agent"])
    summary: str | None = None
    emotional_profile: dict[str, float] = Field(default_factory=dict)
    emotional_valence: float | None = None
    importance: float = Field(default=0.5)
    modality: str = "text"
    media: MediaInputModel | None = None
    media_type: Literal["image", "audio", "video", "pdf"] | None = None
    source_mime_type: str | None = None


class RememberProcedureRequest(MCPModel):
    content: str = Field(min_length=1)
    steps: list[str] = Field(min_length=1)
    preconditions: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5)
    modality: str = "text"
    media: MediaInputModel | None = None
    media_type: Literal["image", "audio", "video", "pdf"] | None = None
    text_description: str | None = None

    @field_validator("steps", "preconditions")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("steps and preconditions must contain only non-blank strings")
        return values


class RememberFactResponse(MCPModel):
    record: MemoryRecordModel
    potential_contradictions: list[ContradictionCandidateModel]
    contradiction_check: ContradictionCheckModel
    contradiction_check_order: Literal["after_store"] = "after_store"
    total_contradictions: int
    contradictions_truncated: bool


class RememberEpisodeResponse(MCPModel):
    record: MemoryRecordModel


class RememberProcedureResponse(MCPModel):
    record: MemoryRecordModel


class RecallMemoriesRequest(MCPModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_RETRIEVAL_TOP_K, ge=1, le=MAX_RETRIEVAL_TOP_K)
    memory_types: list[MemoryType] | None = None


class RecallMemoriesResponse(MCPModel):
    results: list[RankedMemoryResultModel]


class RecallRecentEpisodesRequest(MCPModel):
    mode: Literal["recent"]
    limit: int = Field(default=DEFAULT_RECENT_EPISODES, ge=1, le=MAX_RECENT_EPISODES)


class RecallSessionEpisodesRequest(MCPModel):
    mode: Literal["session"]
    session_id: str = Field(min_length=1)


class RecallTimeRangeEpisodesRequest(MCPModel):
    mode: Literal["time_range"]
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_range(self) -> "RecallTimeRangeEpisodesRequest":
        if self.start > self.end:
            raise ValueError("start must be earlier than or equal to end")
        return self


RecallEpisodesRequest = Annotated[
    RecallRecentEpisodesRequest | RecallSessionEpisodesRequest | RecallTimeRangeEpisodesRequest,
    Field(discriminator="mode"),
]
RecallEpisodesRequestAdapter = TypeAdapter(RecallEpisodesRequest)


class RecallEpisodesResponse(MCPModel):
    mode: Literal["recent", "session", "time_range"]
    records: list[MemoryRecordModel]


class RecallProceduresRequest(MCPModel):
    task: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=MAX_RETRIEVAL_TOP_K)


class RecallProceduresResponse(MCPModel):
    ranking_basis: Literal["combined_score"] = "combined_score"
    results: list[ProcedureMatchModel]


class GetMemoryRequest(MCPModel):
    memory_type: MemoryType
    record_id: str = Field(min_length=1)


class GetMemoryResponse(MCPModel):
    record: MemoryRecordModel | None = None


class ResolveContradictionRequest(MCPModel):
    keep_id: str = Field(min_length=1)
    supersede_id: str = Field(min_length=1)


class ResolveContradictionResponse(MCPModel):
    kept_id: str
    superseded_id: str
    status: Literal["resolved"] = "resolved"


class RecordProcedureOutcomeRequest(MCPModel):
    record_id: str = Field(min_length=1)
    success: bool


class RecordProcedureOutcomeResponse(MCPModel):
    record: MemoryRecordModel


class MemoryOverviewModel(MCPModel):
    semantic_count: int
    episodic_count: int
    procedural_count: int
    recent_sessions: list[str]
    latest_events: list[dict[str, Any]]


class ForgettingCycleRequest(MCPModel):
    max_decisions: int = Field(default=DEFAULT_MAX_DECISIONS, ge=1, le=MAX_MAX_DECISIONS)


class ForgettingDecisionModel(MCPModel):
    record_id: str
    memory_type: str
    action: ForgettingAction
    reason: str | None = None
    score: float
    media_deleted: bool = False
    executed: bool = False
    record_skip_reason: str | None = None
    media_skip_reason: str | None = None
    old_importance: float | None = None
    new_importance: float | None = None


class ForgettingReportModel(MCPModel):
    status: Literal["completed", "already_running"]
    dry_run: bool
    scanned: int | None = None
    kept: int | None = None
    faded: int | None = None
    pruned: int | None = None
    media_deleted: int | None = None
    duplicates_flagged: int | None = None
    skipped_records: int | None = None
    skipped_media: int | None = None
    by_type: dict[str, dict[str, int]] | None = None
    decisions: list[ForgettingDecisionModel] = Field(default_factory=list)
    displayed_decisions: int = 0
    total_decisions: int = 0
    decisions_truncated: bool = False


def serialise_record(record: MemoryRecord) -> MemoryRecordModel:
    payload: dict[str, Any] = {
        "id": record.id,
        "memory_type": record.memory_type,
        "modality": record.modality,
        "content": record.content,
        "created_at": record.created_at,
        "last_accessed_at": record.last_accessed_at,
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
                "superseded_by": record.superseded_by,
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
    return MemoryRecordModel.model_validate(payload)


def serialise_ranked_result(result: RankedResult) -> RankedMemoryResultModel:
    return RankedMemoryResultModel(
        record=serialise_record(result.record),
        raw_similarity=result.raw_similarity,
        recency_score=result.recency_score,
        importance_score=result.importance_score,
        final_score=result.final_score,
    )


def serialise_procedural_match(match: ProceduralMatch) -> ProcedureMatchModel:
    return ProcedureMatchModel(
        record=serialise_record(match.record),
        similarity=match.similarity,
        wilson_score=match.wilson_score,
        combined_score=match.combined_score,
    )


def serialise_contradiction_candidate(
    candidate: ContradictionCandidate,
) -> ContradictionCandidateModel:
    return ContradictionCandidateModel(
        record=serialise_record(candidate.record),
        similarity=candidate.similarity,
    )


def serialise_remember_fact_response(
    *,
    record: SemanticMemory,
    contradiction_status: ContradictionStatus,
    contradiction_candidates: list[ContradictionCandidate],
    contradiction_detail: str | None = None,
    max_contradictions: int = DEFAULT_MAX_CONTRADICTIONS,
) -> RememberFactResponse:
    bounded_limit = min(max_contradictions, MAX_MAX_CONTRADICTIONS)
    displayed = contradiction_candidates[:bounded_limit]
    return RememberFactResponse(
        record=serialise_record(record),
        potential_contradictions=[
            serialise_contradiction_candidate(candidate) for candidate in displayed
        ],
        contradiction_check=ContradictionCheckModel(
            status=contradiction_status,
            detail=contradiction_detail,
        ),
        total_contradictions=len(contradiction_candidates),
        contradictions_truncated=len(contradiction_candidates) > len(displayed),
    )


def serialise_forgetting_report(
    report: ForgettingReport,
    *,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
) -> ForgettingReportModel:
    bounded_limit = min(max_decisions, MAX_MAX_DECISIONS)
    decisions = report.decisions[:bounded_limit]
    return ForgettingReportModel(
        status="completed",
        dry_run=report.dry_run,
        scanned=report.scanned,
        kept=report.kept,
        faded=report.faded,
        pruned=report.pruned,
        media_deleted=report.media_deleted,
        duplicates_flagged=report.duplicates_flagged,
        skipped_records=report.skipped_records,
        skipped_media=report.skipped_media,
        by_type=dict(report.by_type),
        decisions=[
            ForgettingDecisionModel(
                record_id=decision.record_id,
                memory_type=decision.memory_type,
                action=decision.action,
                reason=decision.reason,
                score=decision.score,
                media_deleted=decision.media_deleted,
                executed=decision.executed,
                record_skip_reason=decision.record_skip_reason,
                media_skip_reason=decision.media_skip_reason,
                old_importance=decision.old_importance,
                new_importance=decision.new_importance,
            )
            for decision in decisions
        ],
        displayed_decisions=len(decisions),
        total_decisions=len(report.decisions),
        decisions_truncated=len(report.decisions) > len(decisions),
    )


def serialise_memory_overview(payload: dict[str, Any]) -> MemoryOverviewModel:
    return MemoryOverviewModel.model_validate(payload)


def get_memory_by_type(service: MemoryRuntime, request: GetMemoryRequest) -> MemoryRecord | None:
    store = {
        "semantic": service.semantic_store,
        "episodic": service.episodic_store,
        "procedural": service.procedural_store,
    }[request.memory_type]
    return store.get_by_id(request.record_id)
