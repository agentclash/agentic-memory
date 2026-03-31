"""Agentic Memory SDK — a clean Python interface over the cognitive memory runtime.

Usage::

    from agentic_memory import Memory

    m = Memory()
    m.remember("user prefers dark mode")
    m.remember_episode("debugged auth flow for 2 hours", session="abc")
    results = m.recall("what does the user prefer?")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from forgetting.contradiction import ContradictionCandidate
from forgetting.service import ForgettingReport
from models.base import MemoryRecord
from models.episodic import EpisodicMemory
from models.procedural import ProceduralMemory
from models.semantic import SemanticMemory
from retrieval.ranking import RankedResult
from runtime import MemoryRuntime, build_runtime
from stores.procedural_store import ProceduralMatch
from utils.embeddings import TextEmbedder

__all__ = [
    "Memory",
    "MemoryRuntime",
    "RankedResult",
    "ContradictionCandidate",
    "ProceduralMatch",
    "ForgettingReport",
    "SemanticMemory",
    "EpisodicMemory",
    "ProceduralMemory",
    "MemoryRecord",
]


class Memory:
    """High-level interface to the cognitive memory system.

    Wraps :func:`runtime.build_runtime` and exposes remember / recall /
    forget operations with minimal boilerplate.

    Parameters
    ----------
    chroma_path:
        Directory for ChromaDB persistence.  Defaults to ``./chroma_db``.
    media_root:
        Root directory for owned media files.  Defaults to ``./data/media``.
    embedder:
        Custom embedder instance.  Defaults to :class:`GeminiEmbedder`.
    embedding_dimensions:
        Vector dimensionality (must match *embedder* if both are given).
    max_media_bytes:
        Per-file size cap for media embedding.

    Examples
    --------
    >>> m = Memory()
    >>> m.remember("the deploy key rotates every 90 days")
    'some-uuid'
    >>> m.recall("deploy key rotation")
    [RankedResult(...)]
    """

    def __init__(
        self,
        *,
        chroma_path: str | None = None,
        media_root: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        embedding_dimensions: int | None = None,
        max_media_bytes: int | None = None,
    ) -> None:
        self._runtime = build_runtime(
            chroma_path=chroma_path,
            media_root=media_root,
            embedder=embedder,
            embedding_dimensions=embedding_dimensions,
            max_media_bytes=max_media_bytes,
        )

    @property
    def runtime(self) -> MemoryRuntime:
        """Escape hatch — direct access to the underlying runtime components."""
        return self._runtime

    # ── remember ──────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        *,
        importance: float = 0.5,
        category: str = "general",
        domain: str | None = None,
        confidence: float = 1.0,
        supersedes: str | None = None,
        related_ids: list[str] | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        media_ref: str | None = None,
        media_type: str | None = None,
        text_description: str | None = None,
    ) -> str:
        """Store a semantic memory (fact, preference, concept).

        Returns the record ID.
        """
        record = SemanticMemory(
            content=content,
            importance=importance,
            category=category,
            domain=domain,
            confidence=confidence,
            supersedes=supersedes,
            related_ids=related_ids or [],
            source=source,
            metadata=metadata or {},
            media_ref=media_ref,
            media_type=media_type,
            text_description=text_description,
            modality="text" if not media_ref else (media_type or "text"),
        )
        return self._runtime.semantic_store.store(record)

    def remember_episode(
        self,
        content: str,
        *,
        session: str | None = None,
        turn: int | None = None,
        participants: list[str] | None = None,
        summary: str | None = None,
        emotional_valence: float | None = None,
        emotional_profile: dict[str, float] | None = None,
        importance: float = 0.5,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        media_ref: str | None = None,
        media_type: str | None = None,
        text_description: str | None = None,
    ) -> str:
        """Store an episodic memory (event, interaction, experience).

        Returns the record ID.
        """
        kwargs: dict[str, Any] = dict(
            content=content,
            importance=importance,
            turn_number=turn,
            participants=participants or ["user", "agent"],
            summary=summary,
            emotional_valence=emotional_valence,
            emotional_profile=emotional_profile or {},
            source=source,
            metadata=metadata or {},
            media_ref=media_ref,
            media_type=media_type,
            text_description=text_description,
            modality="text" if not media_ref else (media_type or "text"),
        )
        if session is not None:
            kwargs["session_id"] = session
        record = EpisodicMemory(**kwargs)
        return self._runtime.episodic_store.store(record)

    def remember_procedure(
        self,
        content: str,
        *,
        steps: list[str],
        preconditions: list[str] | None = None,
        importance: float = 0.5,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        media_ref: str | None = None,
        media_type: str | None = None,
        text_description: str | None = None,
    ) -> str:
        """Store a procedural memory (skill, workflow, strategy).

        Returns the record ID.
        """
        record = ProceduralMemory(
            content=content,
            steps=steps,
            preconditions=preconditions or [],
            importance=importance,
            source=source,
            metadata=metadata or {},
            media_ref=media_ref,
            media_type=media_type,
            text_description=text_description,
            modality="text" if not media_ref else (media_type or "text"),
        )
        return self._runtime.procedural_store.store(record)

    # ── recall ────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        types: list[str] | None = None,
        relevance_weight: float = 0.4,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
    ) -> list[RankedResult]:
        """Search across memory stores with weighted ranking.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Maximum results to return.
        types:
            Filter to specific memory types (``"semantic"``, ``"episodic"``,
            ``"procedural"``).  ``None`` searches all.
        relevance_weight / recency_weight / importance_weight:
            Ranking weights (should sum to 1.0).
        """
        return self._runtime.retriever.query(
            query,
            top_k=top_k,
            memory_types=types,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
        )

    def recall_by_vector(
        self,
        vector: list[float],
        *,
        top_k: int = 5,
        types: list[str] | None = None,
        relevance_weight: float = 0.4,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
    ) -> list[RankedResult]:
        """Search by pre-computed embedding vector."""
        return self._runtime.retriever.query_by_vector(
            vector,
            top_k=top_k,
            memory_types=types,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
        )

    def recall_episodes(
        self,
        *,
        mode: str = "recent",
        limit: int = 5,
        session: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[MemoryRecord]:
        """Retrieve episodic memories by recency, session, or time range.

        Parameters
        ----------
        mode:
            ``"recent"`` — most recent episodes (use *limit*).
            ``"session"`` — all episodes in a session (use *session*).
            ``"time_range"`` — episodes within a window (use *start* and *end*).
        """
        if mode == "recent":
            return self._runtime.retriever.query_recent(limit)
        if mode == "session":
            if session is None:
                raise ValueError("session is required when mode='session'")
            return self._runtime.episodic_store.get_by_session(session)
        if mode == "time_range":
            if start is None or end is None:
                raise ValueError("start and end are required when mode='time_range'")
            return self._runtime.retriever.query_time_range(start, end)
        raise ValueError(f"Unknown mode '{mode}'. Use 'recent', 'session', or 'time_range'.")

    def recall_procedures(
        self,
        task: str,
        *,
        top_k: int = 3,
    ) -> list[ProceduralMatch]:
        """Find the best procedures for a task, ranked by similarity + Wilson score."""
        return self._runtime.procedural_store.get_best_procedure_matches(task, top_k=top_k)

    # ── contradictions ────────────────────────────────────────────────────

    def find_contradictions(
        self,
        record_id: str,
        *,
        threshold: float = 0.85,
        top_k: int = 5,
    ) -> list[ContradictionCandidate]:
        """Find semantic memories that may contradict a stored record."""
        record = self._runtime.semantic_store.get_by_id(record_id)
        if record is None:
            raise ValueError(f"Semantic record '{record_id}' not found")
        return self._runtime.contradiction_detector.find_potential_contradictions(
            record, threshold=threshold, top_k=top_k
        )

    def resolve_contradiction(
        self,
        *,
        keep_id: str,
        supersede_id: str,
    ) -> None:
        """Mark one semantic memory as superseding another."""
        self._runtime.contradiction_detector.resolve_supersession(
            superseded_id=supersede_id, kept_id=keep_id
        )

    # ── procedural outcomes ───────────────────────────────────────────────

    def record_outcome(self, record_id: str, *, success: bool) -> None:
        """Record a success or failure for a procedural memory."""
        self._runtime.procedural_store.record_outcome(record_id, success)

    # ── forgetting ────────────────────────────────────────────────────────

    def forget(self, *, dry_run: bool = False) -> ForgettingReport:
        """Run the forgetting cycle.

        With ``dry_run=True``, returns what *would* happen without modifying
        any records.
        """
        return self._runtime.forgetting_service.run_cycle(dry_run=dry_run)

    # ── direct access ─────────────────────────────────────────────────────

    def get(self, record_id: str, *, type: str) -> MemoryRecord | None:
        """Fetch a single record by ID and type.

        Parameters
        ----------
        type:
            One of ``"semantic"``, ``"episodic"``, ``"procedural"``.
        """
        store_map = {
            "semantic": self._runtime.semantic_store,
            "episodic": self._runtime.episodic_store,
            "procedural": self._runtime.procedural_store,
        }
        store = store_map.get(type)
        if store is None:
            raise ValueError(f"Unknown type '{type}'. Use 'semantic', 'episodic', or 'procedural'.")
        return store.get_by_id(record_id)

    def overview(self) -> dict[str, Any]:
        """Return a summary of the memory system state."""
        semantic_count = len(self._runtime.semantic_store.get_all_records())
        episodic_count = len(self._runtime.episodic_store.get_all_records())
        procedural_count = len(self._runtime.procedural_store.get_all_records())
        return {
            "total": semantic_count + episodic_count + procedural_count,
            "semantic": semantic_count,
            "episodic": episodic_count,
            "procedural": procedural_count,
        }

    def events(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent lifecycle events (stored, retrieved, faded, pruned, etc.)."""
        return self._runtime.events.snapshot(limit=limit)
